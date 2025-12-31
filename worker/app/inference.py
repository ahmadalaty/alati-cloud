# /worker/app/inference.py
import os
import json
from io import BytesIO
from typing import Dict, Tuple

import torch
import torchvision.transforms as T
from PIL import Image

# IMPORTANT: this must match your storage.py helper
# (your worker must have the same storage.py as backend)
from .storage import load_bytes_from_ref
from .config import settings

# Build marker so you can prove the worker updated
BUILD_MARKER = "WORKER_INFERENCE_R2_BYTES_v1_2026_01_01"

# -------------------------
# Paths / labels
# -------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model_files")

LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    LABELS = json.load(f)

NUM_CLASSES = len(LABELS)

DEFAULT_VARIANT = os.getenv("MODEL_VARIANT", "resnet18").strip().lower()
TRIAGE_THRESHOLD = float(os.getenv("TRIAGE_THRESHOLD", "0.5"))


# -------------------------
# Helpers
# -------------------------
def _clean_state_dict(state: dict) -> dict:
    """
    Converts keys like:
      'module.backbone.conv1.weight' -> 'conv1.weight'
      'backbone.layer1.0...' -> 'layer1.0...'
      'model.layer1.0...' -> 'layer1.0...'
    """
    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("backbone."):
            nk = nk[len("backbone.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    return cleaned


def _extract_state_dict(ckpt):
    """
    Handles:
    - state_dict directly (dict of tensors / OrderedDict)
    - dict containing "state_dict" or "model_state_dict"
    """
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
        return ckpt  # already state dict / OrderedDict
    return None


# -------------------------
# Model loading
# -------------------------
def load_model(model_variant: str):
    model_variant = (model_variant or "resnet18").strip().lower()

    if model_variant == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(weights=None)
        weights_path = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet50.pth")
        active_variant = "resnet50"
    else:
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        weights_path = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet18.pth")
        active_variant = "resnet18"

    # set classifier head
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

    ckpt = torch.load(weights_path, map_location="cpu")

    # if someone saved full model object (rare)
    if not isinstance(ckpt, dict):
        ckpt.eval()
        return ckpt, active_variant

    state = _extract_state_dict(ckpt)
    if state is None:
        raise RuntimeError("Checkpoint format not understood (no state_dict found)")

    state = _clean_state_dict(state)

    # strict first
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        # allow partial (head mismatch, etc.)
        model.load_state_dict(state, strict=False)

    model.eval()
    return model, active_variant


MODEL, ACTIVE_VARIANT = load_model(DEFAULT_VARIANT)


# -------------------------
# Preprocessing
# -------------------------
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def _normalize_eye(eye: str) -> str:
    e = (eye or "").strip().lower()
    if e in ("l", "left"):
        return "left"
    if e in ("r", "right"):
        return "right"
    return "left"


def _infer_storage_mode_from_ref(ref: str) -> str:
    """
    If STORAGE_MODE=r2, always treat ref as an R2 key.
    Otherwise treat it as local path.
    """
    mode = (settings.STORAGE_MODE or "local").strip().lower()
    if mode == "r2":
        return "r2"
    return "local"


# -------------------------
# Inference (R2-first)
# -------------------------
def run_inference(image_ref: str, eye: str):
    """
    image_ref:
      - if STORAGE_MODE=r2: the R2 key e.g. "uploads/<uuid>.jpg"
      - else: local file path

    Returns dict safe for JSON
    """
    eye = _normalize_eye(eye)

    mode = _infer_storage_mode_from_ref(image_ref)

    # Load bytes from R2 or local
    img_bytes = load_bytes_from_ref(mode, image_ref)

    # Open as PIL image from bytes
    img = Image.open(BytesIO(img_bytes)).convert("RGB")

    x = TRANSFORM(img).unsqueeze(0)

    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.sigmoid(logits)[0]

    probs_dict = {LABELS[i]: float(probs[i].item()) for i in range(NUM_CLASSES)}
    top_label = max(probs_dict, key=probs_dict.get)
    top_prob = float(probs_dict[top_label])
    triage = "Refer" if top_prob >= TRIAGE_THRESHOLD else "Routine"

    return {
        "model_variant": ACTIVE_VARIANT,
        "eye": eye,
        "probs": probs_dict,
        "top_label": top_label,
        "top_prob": top_prob,
        "triage": triage,
    }
