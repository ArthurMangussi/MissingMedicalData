import sys
import os
import base64
import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

app = Flask(__name__)
FEEDBACK_PATH = BASE_DIR / "feedback.json"

# --- Try importing project modules ---
try:
    from codes.data_amputation import ImageDataAmputation

    _AMP = ImageDataAmputation()
    AMPUTATION_AVAILABLE = True
except Exception as e:
    print(f"[MedInpainter] data_amputation not loaded: {e}")
    _AMP = None
    AMPUTATION_AVAILABLE = False

try:
    from utils.MyDataset import Datasets

    DATASETS_AVAILABLE = True
except Exception as e:
    print(f"[MedInpainter] Datasets not loaded: {e}")
    DATASETS_AVAILABLE = False

try:
    from utils.MyModels import ModelsImputation

    MODELS_AVAILABLE = True
except Exception as e:
    print(f"[MedInpainter] ModelsImputation not loaded: {e}")
    MODELS_AVAILABLE = False

# In-memory caches
_img_cache: dict = {}  # dataset_name -> uint8 ndarray (N, 224, 224)
_model_cache: dict = {}  # (algorithm, ...) -> trained imputer

# --- UI Configuration ---
DATASETS = [
    {
        "value": "inbreast",
        "label": "INBreast",
        "description": "Portuguese breast cancer dataset (410 images, BI-RADS labels)",
    },
    {
        "value": "mias",
        "label": "MIAS",
        "description": "Mini-MIAS mammography dataset (~322 images, severity labels)",
    },
    {
        "value": "cbis-ddsm",
        "label": "CBIS-DDSM",
        "description": "Cancer-causing Breast Density Study (benign/malignant)",
    },
    {
        "value": "vindr-reduzido",
        "label": "VinDr-Mammo",
        "description": "Vietnamese mammography subset (1,000 images, BI-RADS 1–5)",
    },
]

ALGORITHMS = [
    {
        "value": "knn",
        "label": "k-NN",
        "description": "K-Nearest Neighbors pixel imputation",
    },
    {
        "value": "mc",
        "label": "Matrix Completion",
        "description": "IterativeSVD low-rank matrix completion",
    },
    {
        "value": "vaewl",
        "label": "VAE-WL",
        "description": "Variational Autoencoder with Weighted Loss",
    },
    {
        "value": "mae-vit",
        "label": "MAE-ViT",
        "description": "Masked Autoencoder Vision Transformer",
    },
    {"value": "dip", "label": "DIP", "description": "Deep Image Prior (unsupervised)"},
    {
        "value": "diffusion",
        "label": "Diffusion",
        "description": "Diffusion model inpainting (HuggingFace)",
    },
]

MECHANISMS = [
    {
        "value": "mcar-dead-pixels",
        "label": "MCAR — Dead Pixels",
        "category": "MCAR",
        "description": "Simulates dead/burnt sensor pixels and small cluster defects",
        "params": [
            {
                "id": "p_single",
                "label": "Single pixel prob.",
                "min": 0.001,
                "max": 0.05,
                "step": 0.001,
                "default": 0.02,
            },
            {
                "id": "p_cluster",
                "label": "Cluster prob.",
                "min": 0.001,
                "max": 0.05,
                "step": 0.001,
                "default": 0.01,
            },
            {
                "id": "cluster_size",
                "label": "Cluster size (px)",
                "min": 2,
                "max": 12,
                "step": 1,
                "default": 5,
            },
        ],
    },
    {
        "value": "mar-squares",
        "label": "MAR — Random Squares",
        "category": "MAR",
        "description": "Random square patches simulating local acquisition errors",
        "params": [
            {
                "id": "square_size",
                "label": "Square size (px)",
                "min": 5,
                "max": 60,
                "step": 1,
                "default": 30,
            },
        ],
    },
    {
        "value": "mnar-stripes",
        "label": "MNAR — Column Stripes",
        "category": "MNAR",
        "description": "Vertical stripe artifacts from defective detector columns",
        "params": [
            {
                "id": "frac_bad_cols",
                "label": "Fraction bad cols",
                "min": 0.005,
                "max": 0.15,
                "step": 0.005,
                "default": 0.01,
            },
            {
                "id": "stripe_width",
                "label": "Stripe width (px)",
                "min": 1,
                "max": 15,
                "step": 1,
                "default": 5,
            },
        ],
    },
    {
        "value": "mnar-saturation",
        "label": "MNAR — Saturation Dropout",
        "category": "MNAR",
        "description": "Intensity-proportional dropout modelling detector saturation",
        "params": [
            {
                "id": "alpha",
                "label": "Sigmoid steepness",
                "min": 2.0,
                "max": 15.0,
                "step": 0.5,
                "default": 8.0,
            },
            {
                "id": "threshold",
                "label": "Threshold",
                "min": 0.3,
                "max": 0.9,
                "step": 0.05,
                "default": 0.65,
            },
        ],
    },
]


# --- Helpers ---


def to_b64(arr: np.ndarray) -> str:
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    _, buf = cv2.imencode(".png", arr)
    return base64.b64encode(buf).decode()


def _load_dataset_images(dataset_name: str) -> np.ndarray | None:
    """Load and cache all images for a dataset. Returns uint8 (N, 224, 224) or None on failure."""
    if dataset_name in _img_cache:
        return _img_cache[dataset_name]
    if not DATASETS_AVAILABLE:
        _img_cache[dataset_name] = None
        return None
    try:
        data = Datasets(dataset_name)
        images, _, _ = data.load_data()
        # images shape: (N, 224, 224), dtype uint8
        _img_cache[dataset_name] = images
        print(f"[MedInpainter] Loaded {dataset_name}: {len(images)} images")
        return images
    except Exception as e:
        print(f"[MedInpainter] Failed to load {dataset_name}: {e}")
        _img_cache[dataset_name] = None
        return None


def get_image(dataset_name: str, idx: int) -> tuple[np.ndarray, int | None]:
    """
    Return (float32 [0,1] image of shape (224,224), total_images | None).
    Falls back to a synthetic image when the dataset is unavailable.
    """
    images = _load_dataset_images(dataset_name)
    if images is not None and len(images) > 0:
        img_u8 = images[idx % len(images)]  # (224, 224) uint8
        return img_u8.astype(np.float32) / 255.0, len(images)
    # Synthetic fallback when dataset paths are unavailable
    rng = np.random.default_rng(idx)
    size = 224
    cy, cx = size // 2, size // 2
    Y, X = np.mgrid[:size, :size].astype(np.float32)
    img = np.exp(-0.5 * (((X - cx) / (size * 0.42)) ** 2 + ((Y - cy) / (size * 0.48)) ** 2)) * 0.78
    for sigma, amp in [(2, 0.06), (5, 0.04), (10, 0.03)]:
        noise = rng.standard_normal((size, size)).astype(np.float32)
        img += cv2.GaussianBlur(noise, (0, 0), sigma) * amp
    lx = cx + int(rng.integers(-size // 5, size // 5))
    ly = cy + int(rng.integers(-size // 6, size // 6))
    lr = int(rng.integers(8, 22))
    lesion = np.exp(-0.5 * (((X - lx) / max(1, lr * 0.65)) ** 2 + ((Y - ly) / max(1, lr * 0.65)) ** 2))
    img += lesion.astype(np.float32) * 0.20
    return np.clip(img, 0, 1), None


def apply_mechanism(img: np.ndarray, mechanism: str, params: dict):
    H, W = img.shape

    if AMPUTATION_AVAILABLE:
        # The project's amputation code expects uint8 [0,255] and normalises internally
        batch_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)[np.newaxis]
        try:
            if mechanism == "mcar-dead-pixels":
                _, missing, mask = _AMP.generate_mcar_dead_pixels(
                    batch_u8,
                    p_single=float(params.get("p_single", 0.02)),
                    p_cluster=float(params.get("p_cluster", 0.01)),
                    cluster_size=int(params.get("cluster_size", 5)),
                )
            elif mechanism == "mar-squares":
                _, missing, mask = _AMP.generate_squares_mask(
                    batch_u8, square_size=int(params.get("square_size", 30))
                )
            elif mechanism == "mnar-stripes":
                _, missing, mask = _AMP.generate_stripes(
                    batch_u8,
                    frac_bad_cols=float(params.get("frac_bad_cols", 0.01)),
                    stripe_width=int(params.get("stripe_width", 5)),
                )
            elif mechanism == "mnar-saturation":
                _, missing, mask = _AMP.generate_mnar_saturation(
                    batch_u8,
                    alpha=float(params.get("alpha", 8.0)),
                    threshold=float(params.get("threshold", 0.65)),
                )
            else:
                missing = batch_u8.astype(np.float32) / 255.0
                mask = np.zeros((1, H, W), np.float32)
            # Squeeze batch + channel dims; missing is already float32 [0,1] from _normalize_and_prepare
            m0 = missing[0]
            if m0.ndim == 3:
                m0 = m0[..., 0]
            k0 = mask[0]
            if k0.ndim == 3:
                k0 = k0[..., 0]
            return m0.astype(np.float32), k0.astype(np.float32)
        except Exception as e:
            print(f"[MedInpainter] Amputation error: {e}")

    # Fallback numpy implementation
    rng = np.random.default_rng()
    mask = np.zeros((H, W), np.float32)

    if mechanism == "mcar-dead-pixels":
        ps = float(params.get("p_single", 0.02))
        pc = float(params.get("p_cluster", 0.01))
        cs = int(params.get("cluster_size", 5))
        mask |= rng.random((H, W)) < ps
        for y, x in np.argwhere(rng.random((H, W)) < pc):
            mask[
                max(0, y - cs // 2) : min(H, y + cs // 2 + 1),
                max(0, x - cs // 2) : min(W, x + cs // 2 + 1),
            ] = 1.0

    elif mechanism == "mar-squares":
        sq = int(params.get("square_size", 30))
        n = max(3, int(H * W * 0.15 / (sq * sq)))
        for _ in range(n):
            y, x = rng.integers(0, H - sq), rng.integers(0, W - sq)
            mask[y : y + sq, x : x + sq] = 1.0

    elif mechanism == "mnar-stripes":
        frac = float(params.get("frac_bad_cols", 0.01))
        sw = int(params.get("stripe_width", 5))
        for c in rng.choice(W, max(1, int(W * frac)), replace=False):
            mask[:, max(0, c - sw // 2) : min(W, c + sw // 2 + 1)] = 1.0

    elif mechanism == "mnar-saturation":
        alpha = float(params.get("alpha", 8.0))
        thr = float(params.get("threshold", 0.65))
        prob = 1.0 / (1.0 + np.exp(-alpha * (img.astype(np.float64) - thr)))
        mask = (rng.random((H, W)) < prob).astype(np.float32)

    corrupted = img.copy()
    corrupted[mask > 0.5] = np.nan
    return corrupted, mask


def _cv2_fallback(corrupted: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """OpenCV TELEA inpainting — used only when ModelsImputation is unavailable."""
    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    img_u8 = (np.nan_to_num(corrupted, nan=0.0) * 255).astype(np.uint8)
    try:
        result = cv2.inpaint(img_u8, mask_u8, 5, cv2.INPAINT_TELEA)
        return result.astype(np.float32) / 255.0
    except Exception:
        return np.nan_to_num(corrupted, nan=0.0).astype(np.float32)


def _apply_batch(images_u8: np.ndarray, mechanism: str, params: dict):
    """Apply missing mechanism to a uint8 batch (N,H,W). Returns (corrupted_float, mask_float) both (N,H,W)."""
    if AMPUTATION_AVAILABLE:
        try:
            if mechanism == "mcar-dead-pixels":
                _, missing, mask = _AMP.generate_mcar_dead_pixels(
                    images_u8,
                    p_single=float(params.get("p_single", 0.02)),
                    p_cluster=float(params.get("p_cluster", 0.01)),
                    cluster_size=int(params.get("cluster_size", 5)),
                )
            elif mechanism == "mar-squares":
                _, missing, mask = _AMP.generate_squares_mask(
                    images_u8, square_size=int(params.get("square_size", 30))
                )
            elif mechanism == "mnar-stripes":
                _, missing, mask = _AMP.generate_stripes(
                    images_u8,
                    frac_bad_cols=float(params.get("frac_bad_cols", 0.01)),
                    stripe_width=int(params.get("stripe_width", 5)),
                )
            elif mechanism == "mnar-saturation":
                _, missing, mask = _AMP.generate_mnar_saturation(
                    images_u8,
                    alpha=float(params.get("alpha", 8.0)),
                    threshold=float(params.get("threshold", 0.65)),
                )
            else:
                return images_u8.astype(np.float32) / 255.0, np.zeros(
                    images_u8.shape, np.float32
                )
            if missing.ndim == 4:
                missing = missing[..., 0] if missing.shape[-1] == 1 else missing[:, 0]
                mask = mask[..., 0] if mask.ndim == 4 and mask.shape[-1] == 1 else mask
            return missing.astype(np.float32), mask.astype(np.float32)
        except Exception as e:
            print(f"[MedInpainter] _apply_batch error: {e}")
    # per-image fallback
    cs, ms = [], []
    for img in images_u8:
        c, m = apply_mechanism(img.astype(np.float32) / 255.0, mechanism, params)
        cs.append(c)
        ms.append(m)
    return np.array(cs), np.array(ms)


def _extract(arr: np.ndarray) -> np.ndarray:
    """Return a single (H,W) float32 [0,1] image from a batch result of any shape."""
    if arr.ndim == 4:  # (N, H, W, C) or (N, C, H, W)
        arr = arr[0]
        if arr.shape[-1] == 1:  # (H, W, 1)
            arr = arr[..., 0]
        elif arr.ndim == 3 and arr.shape[0] == 1:  # (1, H, W)
            arr = arr[0]
    elif arr.ndim == 3:  # (N, H, W)
        arr = arr[0]
    return np.clip(arr.astype(np.float32), 0, 1)


def inpaint_image(
    corrupted: np.ndarray,
    mask: np.ndarray,
    algorithm: str,
    dataset_name: str = None,
    mechanism: str = None,
    params: dict = None,
) -> np.ndarray:
    """
    Inpaint a single image following the ModelsImputation pattern from the
    experimental scripts. Falls back to OpenCV TELEA when ModelsImputation
    is unavailable.
    """
    params = params or {}

    if not MODELS_AVAILABLE:
        return _cv2_fallback(corrupted, mask)

    # Wrap single image into batch — shape (1, H, W) with NaN = missing
    x_test_md = corrupted[np.newaxis]  # (1, 224, 224)
    missing_mask_test = mask[np.newaxis]  # (1, 224, 224)

    try:
        mi = ModelsImputation()

        # ── MAE-ViT / MAE-ViT-GAN ──────────────────────────────────────────
        if algorithm in ("mae-vit", "mae-vit-gan"):
            if algorithm not in _model_cache:
                _model_cache[algorithm] = mi.choose_model(model=algorithm)
            imputer = _model_cache[algorithm]
            result = mi.mae_imputer_transform(
                model=imputer,
                x_test_md_np=x_test_md,
                missing_mask_test_np=missing_mask_test,
            )

        # ── Matrix Completion ───────────────────────────────────────────────
        elif algorithm == "mc":
            imputer = mi.choose_model(model="mc")
            result = imputer.transform(x_test_md, missing_mask_test)

        # ── Diffusion ───────────────────────────────────────────────────────
        elif algorithm == "diffusion":
            if "diffusion" not in _model_cache:
                _model_cache["diffusion"] = mi.choose_model(model="diffusion")
            imputer = _model_cache["diffusion"]
            prompt = (
                "Full-field digital mammography, high-quality breast parenchyma, "
                "no artifacts, no lesions, inpainting task."
            )
            result = mi.diffusion_transform(
                model=imputer,
                x_test_md_np=x_test_md,
                missing_mask_test_np=missing_mask_test,
                prompt=prompt,
                num_inference_steps=150,
            )

        # ── Deep Image Prior ────────────────────────────────────────────────
        elif algorithm == "dip":
            imputer = mi.choose_model(model="dip")
            x_sq = np.expand_dims(x_test_md, axis=-1)  # (1, H, W, 1)
            m_sq = np.expand_dims(missing_mask_test, axis=-1)  # (1, H, W, 1)
            result = imputer.fit_and_transform(
                x_sq.transpose(0, 3, 1, 2),  # → (1, 1, H, W)
                m_sq.transpose(0, 3, 1, 2),  # → (1, 1, H, W)
            ).transpose(
                0, 2, 3, 1
            )  # → (1, H, W, 1)

        # ── VAE with Weighted Loss ──────────────────────────────────────────
        elif algorithm == "vaewl":
            cache_key = ("vaewl", dataset_name, mechanism, str(sorted(params.items())))
            if cache_key not in _model_cache:
                images = _load_dataset_images(dataset_name)
                if images is None:
                    return _cv2_fallback(corrupted, mask)
                n_train = max(1, int(len(images) * 0.8))
                x_train_u8 = images[:n_train]
                x_val_u8 = images[n_train:]
                x_train_clean = x_train_u8.astype(np.float32) / 255.0
                x_val_clean = x_val_u8.astype(np.float32) / 255.0
                x_train_md_arr, _ = _apply_batch(x_train_u8, mechanism, params)
                x_val_md_arr, _ = _apply_batch(x_val_u8, mechanism, params)
                _model_cache[cache_key] = mi.choose_model(
                    model="vaewl",
                    x_train=x_train_clean,
                    x_train_md=x_train_md_arr,
                    x_val=x_val_clean,
                    x_val_md=x_val_md_arr,
                )
            imputer = _model_cache[cache_key]
            result = imputer.transform(x_test_md)

        # ── KNN (and any other algorithm with simple .transform) ────────────
        else:
            imputer = mi.choose_model(model=algorithm)
            result = imputer.transform(x_test_md)

        return _extract(result)

    except Exception as e:
        print(f"[MedInpainter] Inpainting failed ({algorithm}): {e}")
        import traceback

        traceback.print_exc()
        return _cv2_fallback(corrupted, mask)


def compute_metrics(
    original: np.ndarray, imputed: np.ndarray, mask: np.ndarray
) -> dict:
    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr_fn
        from skimage.metrics import structural_similarity as ssim_fn

        mb = mask > 0.5
        if mb.sum() == 0:
            return {}
        mae = float(np.mean(np.abs(original[mb] - imputed[mb])))
        psnr = float(psnr_fn(original, imputed, data_range=1.0))
        ssim = float(ssim_fn(original, imputed, data_range=1.0))
        return {"psnr": round(psnr, 2), "ssim": round(ssim, 4), "mae": round(mae, 5)}
    except Exception:
        return {}


# --- Routes ---


@app.route("/")
def index():
    return render_template(
        "index.html",
        datasets=DATASETS,
        algorithms=ALGORITHMS,
        mechanisms=MECHANISMS,
        amp_available=AMPUTATION_AVAILABLE,
    )


@app.route("/api/process", methods=["POST"])
def process():
    try:
        body = request.get_json(force=True)
        dataset = body.get("dataset", "inbreast")
        mechanism = body.get("mechanism", "mcar-dead-pixels")
        algorithm = body.get("algorithm", "knn")
        params = body.get("params", {})
        idx = int(body.get("seed", 0))

        original, n_images = get_image(dataset, idx)
        corrupted, mask = apply_mechanism(original, mechanism, params)
        corrupted_vis = np.nan_to_num(corrupted, nan=0.0).astype(np.float32)
        imputed = inpaint_image(
            corrupted,
            mask,
            algorithm=algorithm,
            dataset_name=dataset,
            mechanism=mechanism,
            params=params,
        )
        metrics = compute_metrics(original, imputed, mask)

        return jsonify(
            {
                "original": to_b64(original),
                "corrupted": to_b64(corrupted_vis),
                "mask": to_b64(mask),
                "imputed": to_b64(imputed),
                "metrics": metrics,
                "missing_pct": round(float(mask.mean()) * 100, 1),
                "n_images": n_images,
                "image_idx": idx % n_images if n_images else idx,
                "amp_backend": AMPUTATION_AVAILABLE,
            }
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback", methods=["GET"])
def get_feedback():
    data = json.loads(FEEDBACK_PATH.read_text()) if FEEDBACK_PATH.exists() else []
    return jsonify(data)


@app.route("/api/feedback", methods=["POST"])
def post_feedback():
    body = request.get_json(force=True)
    entry = {
        "id": str(uuid.uuid4()),
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": body.get("dataset", ""),
        "mechanism": body.get("mechanism", ""),
        "algorithm": body.get("algorithm", ""),
        "rating": int(body.get("rating", 0)),
        "notes": body.get("notes", ""),
        "metrics": body.get("metrics", {}),
        "seed": body.get("seed", 0),
    }
    data = json.loads(FEEDBACK_PATH.read_text()) if FEEDBACK_PATH.exists() else []
    data.append(entry)
    FEEDBACK_PATH.write_text(json.dumps(data, indent=2))
    return jsonify({"ok": True, "id": entry["id"]})


@app.route("/api/feedback/<entry_id>", methods=["DELETE"])
def delete_feedback(entry_id):
    if not FEEDBACK_PATH.exists():
        return jsonify({"ok": False}), 404
    data = [e for e in json.loads(FEEDBACK_PATH.read_text()) if e["id"] != entry_id]
    FEEDBACK_PATH.write_text(json.dumps(data, indent=2))
    return jsonify({"ok": True})


@app.route("/api/feedback/export", methods=["GET"])
def export_feedback():
    data = json.loads(FEEDBACK_PATH.read_text()) if FEEDBACK_PATH.exists() else []
    from flask import Response
    import io

    lines = ["id,ts,dataset,mechanism,algorithm,rating,psnr,ssim,mae,seed,notes"]
    for e in data:
        m = e.get("metrics", {})
        lines.append(
            ",".join(
                str(x)
                for x in [
                    e["id"],
                    e["ts"],
                    e["dataset"],
                    e["mechanism"],
                    e["algorithm"],
                    e["rating"],
                    m.get("psnr", ""),
                    m.get("ssim", ""),
                    m.get("mae", ""),
                    e.get("seed", ""),
                    f'"{e.get("notes","")}"',
                ]
            )
        )
    csv = "\n".join(lines)
    return Response(
        csv,
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=medinpainter_feedback.csv"
        },
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
