#!/usr/bin/env python3
"""
Phase 1 v2 — Multi-model cross-modal safety probing (corrected)

BUGS FIXED vs v1
================

Bug 1 — CRITICAL: Hook point mismatch
  v1 hooked decoder_layers[i].mlp (MLP sub-module output only).
  The attack steers the FULL residual stream (h_in + Attn + MLP).
  These are different vector spaces with different norms (~3-8 vs ~50-200).
  A probe direction calibrated in MLP-delta space is geometrically wrong
  when applied to the full residual stream.

  FIX: Hook the full LlamaDecoderLayer output so probe and attack operate
  in exactly the same space. This is a one-line change in ActivationCollector.

Bug 2 — CRITICAL: Text probe trained without image context
  v1 collected T_harm / T_ben activations via pure text-only forward passes.
  At attack time the model processes [image_tokens (576) | text_tokens].
  Causal attention means every text token attends to all 576 image tokens,
  producing completely different hidden states than text-only mode.
  This explains why steering worked on cyber (text-dominant keywords like
  "hack", "malware") but failed on physical/financial (harmful word is in
  the IMAGE, not the text prompt, so text-only probe misses it entirely).

  FIX: Collect T_harm / T_ben activations using build_image_inputs with a
  NEUTRAL grey image. This gives every sample the same 576-token image
  context prefix that will be present at attack time, so the probe w_l
  is calibrated in the correct (multimodal) activation space.

Bug 3 — MODERATE: Image probe compared different modalities
  v1: image_probe trained on I_harm (image forward) vs T_ben (TEXT forward).
  This taught the probe to detect "is an image present?" not "is this harmful?".
  The cross-modal Jaccard conclusion in the paper is therefore about
  modality detection overlap, not safety-neuron overlap.

  FIX: Train image_probe on I_harm vs I_ben (both with image context).
  Now both classes go through the same forward-pass modality, so the
  probe genuinely learns harm vs. benign within multimodal inputs.

Bug 4 — MINOR: SVM weights used in wrong space by attack
  v1 stored raw SVM coef_[0] from a Pipeline(StandardScaler, LinearSVC).
  The coef_ lives in SCALED feature space. The attack used it directly
  on unscaled activations, distorting the projection direction.

  FIX: Store per-layer scaler mean and std alongside svm_weights.
  The attack must apply: h_scaled = (h - mean) / std, then project,
  then undo scaling. We also expose a convenience function.
  Additionally we now store the full scaler object in the pkl.

Usage:
    python probing_multimodel_v2.py --model llava1.6 --device cuda:0 --n-samples 500

    # Re-use cached activations (skip re-collection):
    python probing_multimodel_v2.py --model llava1.6 --load-cache
"""

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# =============================================================================
# Configuration
# =============================================================================

DATA_ROOT = Path("../dataset")

PLT_STYLE = {
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.titlesize": 13, "figure.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.25,
    "savefig.dpi": 180, "savefig.bbox": "tight",
}
plt.rcParams.update(PLT_STYLE)

COLORS = {
    "T_harm": "#D7263D", "T_ben": "#1B998B",
    "I_harm": "#2E86AB", "I_ben": "#F46036",
    "accent": "#6A0572", "muted": "#7B7B7B",
}

# =============================================================================
# Model loading (unchanged from v1)
# =============================================================================

def find_decoder_layers(model) -> List:
    layers = []
    for m in model.modules():
        if isinstance(m, LlamaDecoderLayer):
            layers.append(m)
    for i, layer in enumerate(layers):
        if not hasattr(layer.self_attn, "layer_id"):
            layer.self_attn.layer_id = i
    layers.sort(key=lambda m: int(m.self_attn.layer_id))
    return layers


def load_model(model_name: str, device: str):
    if model_name == "llava1.5":
        from transformers import LlavaProcessor, LlavaForConditionalGeneration
        model_id = "/models/llava-hf_llava-1.5-7b-hf"
        processor = LlavaProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map={"": device},
            low_cpu_mem_usage=True, attn_implementation="eager")
    elif model_name == "llava1.6":
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        model_id = "/models/llava-hf_llava-v1.6-vicuna-7b-hf"
        processor = LlavaNextProcessor.from_pretrained(model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map={"": device},
            low_cpu_mem_usage=True, attn_implementation="eager")
    elif model_name == "instructblip":
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        model_id = "/models/instructblip-vicuna-7b"
        processor = InstructBlipProcessor.from_pretrained(model_id)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map={"": device},
            low_cpu_mem_usage=True, attn_implementation="eager")
    else:
        raise NotImplementedError(f"Model {model_name} not supported.")
    model.eval()
    decoder_layers = find_decoder_layers(model)
    print(f"  Loaded {model_name}: {len(decoder_layers)} decoder layers")
    return model, processor, decoder_layers


# =============================================================================
# BUG 1 FIX — ActivationCollector hooks FULL DECODER LAYER, not MLP sub-module
# =============================================================================

class ActivationCollector:
    """
    Collect the output of the FULL LlamaDecoderLayer at each layer.

    This captures the complete residual stream:
        h_out = h_in + Attention(LN(h_in)) + MLP(LN(h_in + Attention(LN(h_in))))

    WHY THIS MATTERS:
    The attack's SafetyVectorSteeringAttack hooks the full decoder layer and
    modifies h_out. The probe must train on the same h_out so that the SVM
    weight vector w_l lives in the same geometric space as the attack target.

    v1 hooked decoder_layers[i].mlp which captured only the MLP sub-output
    (the additive delta, norm ~3-8), not the full residual stream (norm ~50-200).
    The resulting w_l was geometrically miscalibrated for attack use.

    TOKEN AGGREGATION:
    We use "last" by default — the last text token position, which is where
    the model's safety decision is made (the position that predicts the first
    output token). For image inputs, we use the last token in the FULL sequence
    including image tokens, which is the final instruction token, not an image
    token. This ensures comparability across text-only and multimodal inputs.
    """

    def __init__(self, decoder_layers: List, num_layers: int,
                 hidden_dim: int, token_agg: str = "last"):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.token_agg  = token_agg
        self._buffer: Dict[int, Optional[torch.Tensor]] = {i: None for i in range(num_layers)}
        self._hooks: List = []

        for i in range(num_layers):
            def make(idx):
                def hook(module, inp, out):
                    # FIX 1: out is (hidden_state, past_kv, ...) for full decoder layer
                    # hidden_state shape: (B, T, d)
                    h = out[0] if isinstance(out, tuple) else out
                    self._buffer[idx] = h.detach().float().cpu()
                return hook
            # FIX 1: hook the FULL decoder layer, not decoder_layers[i].mlp
            self._hooks.append(decoder_layers[i].register_forward_hook(make(i)))

    def pool(self) -> np.ndarray:
        """Return (num_layers, hidden_dim) array for current sample."""
        out = []
        for i in range(self.num_layers):
            a = self._buffer[i]
            if a is None:
                out.append(np.zeros(self.hidden_dim, dtype=np.float32))
                continue
            a = a[0]  # batch=1 -> (T, d)
            if self.token_agg == "max":
                v = a.max(dim=0).values.numpy()
            elif self.token_agg == "mean":
                v = a.mean(dim=0).numpy()
            else:  # "last" — final token position
                v = a[-1].numpy()
            out.append(v.astype(np.float32))
        return np.stack(out, axis=0)  # (L, d)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# =============================================================================
# BUG 2 FIX — All activation collection goes through build_image_inputs
#              with NEUTRAL image for text samples
# =============================================================================

# A single 336x336 neutral grey image shared across all text-mode collections.
# Using grey (128,128,128) rather than white/black to minimize any colour-
# specific feature activation while still providing the full 576-token context.
_NEUTRAL_IMAGE = Image.new("RGB", (336, 336), color=(128, 128, 128))


def build_image_inputs(processor, model_name: str,
                       image: Image.Image, instruction: str, device: str) -> dict:
    """Build multimodal inputs for any image + instruction pair."""
    if "InstructBlip" in type(processor).__name__:
        prompt = instruction
    elif hasattr(processor, "apply_chat_template"):
        conv = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction},
        ]}]
        prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    else:
        prompt = f"USER: <image>\n{instruction}\nASSISTANT:"
    inp = processor(images=image, text=prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in inp.items()}


def collect_activations(
    model, processor, model_name: str, collector: ActivationCollector,
    samples: List[dict], device: str, desc: str,
    use_neutral_image: bool = False,
) -> np.ndarray:
    """
    Collect (N, L, d) activations for a list of samples.

    Parameters
    ----------
    use_neutral_image : bool
        If True, use the neutral grey image for every sample (text probe mode).
        If False, load the image from item["image"] (image probe mode).

    FIX 2: ALL collection now goes through multimodal inputs so every
    activation is produced in the same context (576 image tokens present).
    The only difference between text and image probes is WHICH image is used:
      - Text probe: neutral grey image (same context, removes image content)
      - Image probe: the actual harmful/benign FigStep rendered image
    """
    out = []
    for item in tqdm(samples, desc=desc, ncols=80):
        if use_neutral_image:
            image = _NEUTRAL_IMAGE
        else:
            image = Image.open(item["image"]).convert("RGB")

        inp = build_image_inputs(processor, model_name, image,
                                 item["instruction"], device)
        with torch.no_grad():
            _ = model(**inp)
        out.append(collector.pool())
    return np.stack(out, axis=0)  # (N, L, d)


# =============================================================================
# BUG 3 FIX + BUG 4 FIX — Probe fitting stores scaler for attack use
# =============================================================================

def fit_probe(X_pos: np.ndarray, X_neg: np.ndarray,
              c: float = 0.01, top_k: int = 50) -> dict:
    """
    Fit a LinearSVC probe and return weights, neuron IDs, and scaler.

    FIX 4: We now store the fitted StandardScaler (mean, std) so the attack
    can correctly project activations in the scaled space where w_l is defined:
        h_scaled = (h - scaler_mean) / scaler_std
        projection = h_scaled @ w_hat  (both in scaled space)
        h_steered  = h_scaled - alpha * projection * w_hat
        h_unscaled = h_steered * scaler_std + scaler_mean

    Returns
    -------
    dict with keys:
      cv_accuracy, cv_f1      : cross-validated metrics
      svm_weights             : coef_[0] in scaled feature space  (d,)
      neuron_ids              : top-k neuron indices by |weight|   (k,)
      scaler_mean             : StandardScaler mean                (d,)
      scaler_std              : StandardScaler std                  (d,)
      scaler                  : fitted StandardScaler object
      probe_accuracy          : alias for cv_accuracy (attack compatibility)
    """
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.array([1] * len(X_pos) + [0] * len(X_neg))

    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    LinearSVC(C=c, max_iter=8000, dual=False)),
    ])
    cv_acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
    cv_f1  = cross_val_score(pipe, X, y, cv=skf, scoring="f1")
    pipe.fit(X, y)

    scaler: StandardScaler = pipe.named_steps["scaler"]
    w  = pipe.named_steps["svm"].coef_[0]         # direction in scaled space
    tk = np.argsort(np.abs(w))[::-1][:top_k]      # top-k by magnitude

    return {
        "cv_accuracy":   float(cv_acc.mean()),
        "cv_f1":         float(cv_f1.mean()),
        "probe_accuracy": float(cv_acc.mean()),    # alias used by select_top_layers
        "svm_weights":   w,                        # (d,) in SCALED space
        "neuron_ids":    tk,                        # (top_k,) indices
        "scaler_mean":   scaler.mean_.astype(np.float32),
        "scaler_std":    scaler.scale_.astype(np.float32),
        "scaler":        scaler,                   # full object for attack
    }


def centroid_distance(X1: np.ndarray, X2: np.ndarray) -> float:
    X  = np.concatenate([X1, X2], axis=0)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    return float(np.linalg.norm(Xs[:len(X1)].mean(0) - Xs[len(X1):].mean(0)))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    A, B = set(a.tolist()), set(b.tolist())
    return len(A & B) / max(len(A | B), 1)


# =============================================================================
# Attack-side utility: correct projection in scaled space
# =============================================================================

def make_steering_direction(probe_entry: dict,
                             device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (w_hat_scaled, mean, std) tensors for use in the attack.

    The attack should do:
        h_fp32 = h.float()
        h_scaled = (h_fp32 - mean) / std
        coeff = (h_scaled * w_hat_scaled).sum(-1, keepdim=True)
        h_scaled_steered = h_scaled - alpha * coeff * w_hat_scaled
        h_steered = h_scaled_steered * std + mean
        return h_steered.to(h.dtype)

    This ensures the projection is done in the same geometric space where
    the SVM was trained, then the result is mapped back to activation space.
    """
    w    = torch.tensor(probe_entry["svm_weights"], dtype=torch.float32, device=device)
    mean = torch.tensor(probe_entry["scaler_mean"],  dtype=torch.float32, device=device)
    std  = torch.tensor(probe_entry["scaler_std"],   dtype=torch.float32, device=device)
    w_hat = w / (w.norm() + 1e-9)
    return w_hat, mean, std


# =============================================================================
# Plotting (updated for v2 — image probe now I_harm vs I_ben)
# =============================================================================

def plot_per_layer_lines(out_dir, num_layers, text_probe, image_probe,
                         cdists, overlap, exp_jaccard, model_name):
    L   = np.arange(num_layers)
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

    ax = axes[0]
    ax.plot(L, [text_probe[i]["cv_accuracy"]  for i in L], "o-",
            color=COLORS["T_harm"], lw=2, ms=5, label="Text probe (neutral img ctx)")
    ax.plot(L, [image_probe[i]["cv_accuracy"] for i in L], "s-",
            color=COLORS["I_harm"], lw=2, ms=5, label="Image probe (I_harm vs I_ben)")
    ax.axhline(0.5, color=COLORS["muted"], ls=":", lw=1, label="Chance")
    ax.set_ylabel("5-fold CV accuracy")
    ax.set_title("(a) Per-layer probe accuracy")
    ax.set_ylim(0.45, 1.05)
    ax.legend(loc="lower right", framealpha=0.95)

    ax = axes[1]
    ax.plot(L, cdists["d_Th_Tb"], "o-", color=COLORS["T_harm"], lw=2, ms=5,
            label="d(T_harm, T_ben) [neutral ctx]")
    ax.plot(L, cdists["d_Ih_Ib"], "s-", color=COLORS["I_harm"], lw=2, ms=5,
            label="d(I_harm, I_ben) [image ctx]")
    ax.plot(L, cdists["d_Th_Ih"], "^-", color=COLORS["accent"], lw=2, ms=5,
            label="d(T_harm, I_harm) [cross-modal]")
    ax.set_ylabel("Centroid distance (standardised)")
    ax.set_title("(b) Per-layer centroid distances")
    ax.legend(loc="best", framealpha=0.95)

    ax = axes[2]
    ax.bar(L, overlap, color=COLORS["accent"], edgecolor="white", linewidth=0.6)
    ax.axhline(exp_jaccard, color="red", ls="--", lw=1.2,
               label=f"Random baseline ({exp_jaccard:.3f})")
    ax.axhline(overlap.mean(), color=COLORS["muted"], ls=":", lw=1.2,
               label=f"Mean ({overlap.mean():.3f})")
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("Jaccard (text-probe vs image-probe top-k)")
    ax.set_title("(c) Cross-modal JailNeuron overlap")
    ax.legend(loc="best", framealpha=0.95)

    fig.suptitle(f"{model_name} — per-layer analysis (v2 corrected)", y=0.995)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_per_layer_lines.png")
    plt.close(fig)
    print(f"  Saved fig_per_layer_lines.png")


def plot_neuron_clusters(out_dir, layers_to_plot, acts_dict, num_layers):
    cols = 3
    rows = (len(layers_to_plot) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2*cols, 4.4*rows))
    axes = np.atleast_2d(axes).flatten()

    for ax_idx, li in enumerate(layers_to_plot):
        ax = axes[ax_idx]
        sets, labels = [], []
        for name in ("T_harm", "T_ben", "I_harm", "I_ben"):
            if name in acts_dict and acts_dict[name] is not None:
                sets.append(acts_dict[name][:, li, :])
                labels.append(name)

        X_all = np.concatenate(sets, axis=0)
        sc  = StandardScaler().fit(X_all)
        Xs  = sc.transform(X_all)
        pca = PCA(n_components=2, random_state=42).fit(Xs)

        centroids = {}
        for X, name in zip(sets, labels):
            P = pca.transform(sc.transform(X))
            ax.scatter(P[:,0], P[:,1], s=18, alpha=0.45, color=COLORS[name],
                       edgecolor="white", linewidth=0.4,
                       label=f"{name} (n={len(X)})")
            cen = P.mean(0)
            centroids[name] = cen
            ax.scatter(*cen, s=260, color=COLORS[name], marker="X",
                       edgecolor="black", linewidth=1.6, zorder=10)

        title = f"Layer {li}"
        if "T_harm" in centroids and "I_harm" in centroids:
            d = float(np.linalg.norm(centroids["T_harm"] - centroids["I_harm"]))
            title += f"  d(Th,Ih)={d:.2f}"
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("PC1", fontsize=9)
        ax.set_ylabel("PC2", fontsize=9)
        if ax_idx == 0:
            ax.legend(loc="best", fontsize=8, framealpha=0.95)

    for k in range(len(layers_to_plot), len(axes)):
        axes[k].axis("off")
    fig.suptitle("Per-layer neuron clusters (PCA-2, v2 corrected)", y=1.005)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_neuron_clusters_pca.png")
    plt.close(fig)
    print("  Saved fig_neuron_clusters_pca.png")


def plot_jaccard_heatmap(out_dir, text_probe, image_probe, num_layers, exp_jaccard):
    M = np.zeros((num_layers, num_layers))
    for i in range(num_layers):
        ti = set(text_probe[i]["neuron_ids"].tolist())
        for j in range(num_layers):
            ij = set(image_probe[j]["neuron_ids"].tolist())
            M[i, j] = len(ti & ij) / max(len(ti | ij), 1)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(M, cmap="magma", origin="lower", aspect="equal")
    ax.set_xlabel("Image-probe layer (j)")
    ax.set_ylabel("Text-probe layer (i)")
    ax.plot([0, num_layers-1], [0, num_layers-1],
            color="cyan", lw=0.8, alpha=0.8, ls=":")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Jaccard overlap")
    ax.set_title(f"Jaccard overlap of top-k JailNeurons\n"
                 f"random baseline {exp_jaccard:.3f} | "
                 f"image probe: I_harm vs I_ben (v2)")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_jaccard_heatmap.png")
    plt.close(fig)
    print("  Saved fig_jaccard_heatmap.png")
    return M


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Probing v2 — corrected activation space and probe targets")
    parser.add_argument("--model",    required=True,
                        choices=["llava1.5", "llava1.6", "instructblip"])
    parser.add_argument("--device",   default="cuda:0")
    parser.add_argument("--n-samples", type=int,   default=500)
    parser.add_argument("--top-k",    type=int,   default=50)
    parser.add_argument("--svm-c",    type=float, default=0.01)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--load-cache", action="store_true",
                        help="Load cached activations from previous run")
    parser.add_argument("--no-plots",   action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(f"results/probing_{args.model}_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("\nLoading data...")
    with open(DATA_ROOT / "advbench_harmful_behaviors.json") as f:
        advbench = json.load(f)
    with open(DATA_ROOT / "alpaca_benign500.json") as f:
        alpaca   = json.load(f)
    with open(DATA_ROOT / "advbench_rendered_images.json") as f:
        adv_images = json.load(f)
    with open(DATA_ROOT / "alpaca_rendered_images.json") as f:
        alp_images = json.load(f)

    rng = random.Random(args.seed)
    for lst in [advbench, alpaca, adv_images, alp_images]:
        rng.shuffle(lst)

    N = args.n_samples
    # Text samples: instruction only (image will be added as neutral)
    T_harm_samples = [{"instruction": x["instruction"]} for x in advbench[:N]]
    T_ben_samples  = [{"instruction": x["instruction"]} for x in alpaca[:N]]
    # Image samples: instruction + image path
    I_harm_samples = adv_images[:N]
    I_ben_samples  = alp_images[:N]

    print(f"  T_harm: {len(T_harm_samples)}, T_ben: {len(T_ben_samples)}, "
          f"I_harm: {len(I_harm_samples)}, I_ben: {len(I_ben_samples)}")

    # ── 2. Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading {args.model}...")
    model, processor, decoder_layers = load_model(args.model, args.device)
    model.eval()
    num_layers = len(decoder_layers)
    # hidden_dim from full decoder output = model hidden size
    hidden_dim = decoder_layers[0].self_attn.q_proj.in_features
    print(f"  Layers: {num_layers}, Hidden dim (residual stream): {hidden_dim}")

    # FIX 1: Hook full decoder layer
    collector = ActivationCollector(
        decoder_layers, num_layers, hidden_dim, token_agg="last")

    # ── 3. Collect activations ────────────────────────────────────────────────
    cache_path = out_dir / "activations_v2.pkl"
    if args.load_cache and cache_path.exists():
        print("\nLoading cached activations...")
        with open(cache_path, "rb") as f:
            acts_dict = pickle.load(f)
    else:
        print("\nCollecting activations...")

        # FIX 2: T_harm with NEUTRAL image (same 576-token context as attack time)
        acts_T_harm = collect_activations(
            model, processor, args.model, collector,
            T_harm_samples, args.device, "T_harm (neutral img)",
            use_neutral_image=True)

        # FIX 2: T_ben with NEUTRAL image
        acts_T_ben = collect_activations(
            model, processor, args.model, collector,
            T_ben_samples, args.device, "T_ben (neutral img)",
            use_neutral_image=True)

        # FIX 2+3: I_harm and I_ben both with their actual images
        acts_I_harm = collect_activations(
            model, processor, args.model, collector,
            I_harm_samples, args.device, "I_harm (FigStep images)",
            use_neutral_image=False)

        acts_I_ben = collect_activations(
            model, processor, args.model, collector,
            I_ben_samples, args.device, "I_ben (Alpaca images)",
            use_neutral_image=False)

        acts_dict = {
            "T_harm": acts_T_harm,   # (N, L, d) multimodal with neutral img
            "T_ben":  acts_T_ben,    # (N, L, d) multimodal with neutral img
            "I_harm": acts_I_harm,   # (N, L, d) multimodal with FigStep img
            "I_ben":  acts_I_ben,    # (N, L, d) multimodal with Alpaca img
        }
        with open(cache_path, "wb") as f:
            pickle.dump(acts_dict, f)
        print(f"  Cached to {cache_path}")

    collector.remove()

    # ── 4. Fit probes ─────────────────────────────────────────────────────────
    print("\nFitting probes...")
    text_probe:  Dict[int, dict] = {}
    image_probe: Dict[int, dict] = {}

    for i in tqdm(range(num_layers), desc="Probes", ncols=80):
        # Text probe: T_harm vs T_ben (BOTH with neutral image context)
        text_probe[i] = fit_probe(
            acts_dict["T_harm"][:, i, :],
            acts_dict["T_ben"][:, i, :],
            c=args.svm_c, top_k=args.top_k)

        # FIX 3: Image probe: I_harm vs I_ben (BOTH with real image context)
        image_probe[i] = fit_probe(
            acts_dict["I_harm"][:, i, :],
            acts_dict["I_ben"][:, i, :],
            c=args.svm_c, top_k=args.top_k)

    print(f"  Mean text probe accuracy:  "
          f"{np.mean([text_probe[i]['cv_accuracy']  for i in range(num_layers)]):.4f}")
    print(f"  Mean image probe accuracy: "
          f"{np.mean([image_probe[i]['cv_accuracy'] for i in range(num_layers)]):.4f}")

    # ── 5. Centroid distances ─────────────────────────────────────────────────
    print("\nCalculating centroid distances...")
    cdists: Dict[str, List[float]] = {
        "d_Th_Tb": [],   # text-harm vs text-benign (neutral img ctx)
        "d_Ih_Ib": [],   # image-harm vs image-benign (FIX 3: both multimodal)
        "d_Th_Ih": [],   # text-harm vs image-harm (cross-modal gap — key finding)
    }
    for i in range(num_layers):
        cdists["d_Th_Tb"].append(centroid_distance(
            acts_dict["T_harm"][:, i, :], acts_dict["T_ben"][:, i, :]))
        cdists["d_Ih_Ib"].append(centroid_distance(
            acts_dict["I_harm"][:, i, :], acts_dict["I_ben"][:, i, :]))
        cdists["d_Th_Ih"].append(centroid_distance(
            acts_dict["T_harm"][:, i, :], acts_dict["I_harm"][:, i, :]))
    cdists = {k: np.array(v) for k, v in cdists.items()}

    print(f"  Mean d(T_harm, T_ben): {cdists['d_Th_Tb'].mean():.4f}")
    print(f"  Mean d(I_harm, I_ben): {cdists['d_Ih_Ib'].mean():.4f}")
    print(f"  Mean d(T_harm, I_harm): {cdists['d_Th_Ih'].mean():.4f}  "
          f"← key cross-modal gap")

    # ── 6. Jaccard overlap ────────────────────────────────────────────────────
    print("\nCalculating Jaccard overlap...")
    overlap = np.array([
        jaccard(text_probe[i]["neuron_ids"], image_probe[i]["neuron_ids"])
        for i in range(num_layers)])
    k, D = args.top_k, hidden_dim
    exp_jaccard = (k * k / D) / (2 * k - k * k / D)
    print(f"  Mean Jaccard: {overlap.mean():.4f} | "
          f"Random baseline: {exp_jaccard:.4f} | "
          f"Ratio: {overlap.mean()/exp_jaccard:.2f}x")

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    if not args.no_plots:
        print("\nGenerating figures...")
        plot_per_layer_lines(
            out_dir, num_layers, text_probe, image_probe,
            cdists, overlap, exp_jaccard, args.model)
        layers_to_plot = sorted(
            set([0, 3, 8, 13, 16, 20, 24, 28, num_layers-1])
            & set(range(num_layers)))
        plot_neuron_clusters(out_dir, layers_to_plot, acts_dict, num_layers)
        plot_jaccard_heatmap(
            out_dir, text_probe, image_probe, num_layers, exp_jaccard)

    # ── 8. Save results ───────────────────────────────────────────────────────
    print("\nSaving results...")
    summary = {
        "config": {
            "model":      args.model,
            "n_samples":  args.n_samples,
            "top_k":      args.top_k,
            "svm_c":      args.svm_c,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "version":    "v2",
            "fixes": [
                "Bug1: hooks full decoder layer (residual stream)",
                "Bug2: text probe uses neutral-image multimodal context",
                "Bug3: image probe trains I_harm vs I_ben (same modality)",
                "Bug4: scaler mean/std stored for correct attack projection",
            ],
        },
        "global": {
            "text_probe_acc_mean":  float(np.mean([text_probe[i]["cv_accuracy"]  for i in range(num_layers)])),
            "image_probe_acc_mean": float(np.mean([image_probe[i]["cv_accuracy"] for i in range(num_layers)])),
            "jaccard_mean":          float(overlap.mean()),
            "jaccard_random_baseline": exp_jaccard,
            "d_Th_Ih_mean":          float(cdists["d_Th_Ih"].mean()),
        },
        "per_layer": {
            i: {
                "text_acc":         text_probe[i]["cv_accuracy"],
                "image_acc":        image_probe[i]["cv_accuracy"],
                "d_Th_Tb":          float(cdists["d_Th_Tb"][i]),
                "d_Ih_Ib":          float(cdists["d_Ih_Ib"][i]),
                "d_Th_Ih":          float(cdists["d_Th_Ih"][i]),
                "jaccard":          float(overlap[i]),
                "top_text_neurons": text_probe[i]["neuron_ids"].tolist(),
                "top_image_neurons":image_probe[i]["neuron_ids"].tolist(),
            } for i in range(num_layers)
        },
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  Saved results.json")

    # Save full probe results for attack (includes scaler for FIX 4)
    # Note: scaler objects are not JSON-serialisable so we use pickle
    with open(out_dir / "full_results.pkl", "wb") as f:
        pickle.dump({
            "text_probe":          text_probe,
            "image_probe":         image_probe,
            "centroid_distances":  cdists,
            "jaccard_per_layer":   overlap,
            "acts_dict":           acts_dict,
        }, f)
    print("  Saved full_results.pkl")
    print(f"\nAll done. Results in: {out_dir}")


if __name__ == "__main__":
    main()