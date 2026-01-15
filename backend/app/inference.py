import json
import os
from io import BytesIO
from typing import Dict, Tuple

import torch
import torchvision.transforms as T
from PIL import Image

BUILD_MARKER = "INFERENCE_V2_DIAG_ONLY_WITH_DEBUG_2026_01_15"

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model_files")

LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

# If your labels.json is a list like ["N","D","G","C","A","H","M","O"]
# keep it as-is. If it's a dict, we try to normalize it.
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    _labels_raw = json.load(f)

if isinstance(_labels_raw, dict):
    # try best effort: sort by key if keys are numeric-like, otherwise values order
    try:
        LABELS = [_labels_raw[str(i)] for i in range(len(_labels_raw))]
    except Exception:
        LABELS = list(_labels_raw.values())
else:
    LABELS = list(_labels_raw)

# Hard safety: if labels are not ODIR-8, you'll see it in /debug
ODIR_EXPECTED = ["N", "D", "G", "C", "A", "H", "M", "O"]

LABEL_TO_NAME = {
    "N": "Normal",
    "D": "Diabetic Retinopathy",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "AMD",
    "H": "Hypertension",
    "M": "Myopia",
    "O": "Others",
}

NUM_CLASSES = len(LABELS)
DEFAULT_VARIANT = os.getenv("MODEL_VARIANT", "resnet18").strip().lower()
DEVICE = "cpu"


def _clean_state_dict(state: dict) -> dict:
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
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
        return ckpt
    return None


def load_model(model_variant: str) -> Tuple[torch.nn.Module, str, str]:
    model_variant = (model_variant or "resnet18").strip().lower()

    if model_variant == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(weights=None)
        weights_path = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet50.pth")
        active = "resnet50"
    else:
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        weights_path = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet18.pth")
        active = "resnet18"

    # Set classifier head to NUM_CLASSES
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

    if not os.path.exists(weights_path):
        # This is the #1 reason you get constant "Others"
        raise RuntimeError(f"Model weights file not found: {weights_path}")

    ckpt = torch.load(weights_path, map_location=DEVICE)

    # full model saved
    if not isinstance(ckpt, dict):
        ckpt.eval()
        return ckpt, active, "loaded_full_model_object"

    state = _extract_state_dict(ckpt)
    if state is None:
        raise RuntimeError("Checkpoint format not understood (no state_dict found)")

    state = _clean_state_dict(state)

    # load with fallback non-strict
    try:
        model.load_state_dict(state, strict=True)
        load_mode = "strict"
    except Exception:
        model.load_state_dict(state, strict=False)
        load_mode = "non_strict"

    model.eval()
    return model, active, load_mode


TRANSFORM = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

MODEL, ACTIVE_VARIANT, LOAD_MODE = load_model(DEFAULT_VARIANT)


def _probs_from_bytes(image_bytes: bytes) -> Dict[str, float]:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0)

    with torch.no_grad():
        logits = MODEL(x)
        # ODIR can be multi-label, so sigmoid is fine.
        probs = torch.sigmoid(logits)[0].detach().cpu().tolist()

    # map using LABELS order (this must match training!)
    out = {}
    for i in range(min(len(probs), len(LABELS))):
        out[str(LABELS[i])] = float(probs[i])
    return out


def predict_diagnosis(image_bytes: bytes) -> str:
    probs = _probs_from_bytes(image_bytes)

    if not probs:
        return "Uncertain"

    # choose the max label
    top_label = max(probs, key=probs.get)
    top_prob = probs[top_label]

    # IMPORTANT:
    # If model is uncertain, do NOT default to "Others" (it becomes constant).
    # This prevents “Others” spam when probabilities are flat.
    if top_prob < 0.50:
        return "Uncertain"

    return LABEL_TO_NAME.get(top_label, str(top_label))


def predict_debug(image_bytes: bytes) -> dict:
    probs = _probs_from_bytes(image_bytes)
    if probs:
        top_label = max(probs, key=probs.get)
        top_prob = probs[top_label]
    else:
        top_label, top_prob = None, None

    # show top 3 for debugging only
    top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]

    return {
        "build_marker": BUILD_MARKER,
        "active_variant": ACTIVE_VARIANT,
        "load_mode": LOAD_MODE,
        "labels": LABELS,
        "labels_expected_odir8": ODIR_EXPECTED,
        "top_label": top_label,
        "top_prob": top_prob,
        "top3": top3,
    }
