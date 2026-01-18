import json
import os
from io import BytesIO
from typing import Dict, Tuple

import torch
import torchvision.transforms as T
from PIL import Image

BUILD_MARKER = "INFERENCE_V4_FIX_LABELS_DICT_LIST_2026_01_18"

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model_files")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

# -------------------------
# Load labels correctly (YOUR labels.json format)
# -------------------------
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels_raw = json.load(f)

# YOUR labels.json is list of dicts: [{"code":"N","name":"normal"}, ...]
if isinstance(labels_raw, list) and len(labels_raw) > 0 and isinstance(labels_raw[0], dict):
    LABELS = [x.get("code") for x in labels_raw if isinstance(x, dict) and x.get("code")]
else:
    # fallback for old formats (list[str] or dict)
    if isinstance(labels_raw, dict):
        try:
            LABELS = [labels_raw[str(i)] for i in range(len(labels_raw))]
        except Exception:
            LABELS = list(labels_raw.values())
    else:
        LABELS = list(labels_raw)

LABELS = [str(x).strip() for x in LABELS if str(x).strip()]
NUM_CLASSES = len(LABELS)

LABEL_TO_NAME = {
    "N": "Normal",
    "D": "Diabetic Retinopathy",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "AMD",
    "H": "Hypertensive Retinopathy",
    "M": "Myopia",
    "O": "Other",
}

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

    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

    if not os.path.exists(weights_path):
        raise RuntimeError(f"Model weights not found: {weights_path}")

    ckpt = torch.load(weights_path, map_location=DEVICE)

    # If full model object saved
    if not isinstance(ckpt, dict):
        ckpt.eval()
        return ckpt, active, "full_model"

    state = _extract_state_dict(ckpt)
    if state is None:
        raise RuntimeError("Checkpoint format not understood")

    state = _clean_state_dict(state)

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
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ]
)

MODEL, ACTIVE_VARIANT, LOAD_MODE = load_model(DEFAULT_VARIANT)


def _probs_from_bytes(image_bytes: bytes) -> Dict[str, float]:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0)

    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.sigmoid(logits)[0].detach().cpu().tolist()

    out = {}
    for i in range(min(len(probs), len(LABELS))):
        out[LABELS[i]] = float(probs[i])
    return out


def predict_diagnosis(image_bytes: bytes) -> str:
    probs = _probs_from_bytes(image_bytes)
    if not probs:
        return "Uncertain"

    top_label = max(probs, key=probs.get)
    top_prob = probs[top_label]

    # If low confidence, return uncertain (prevents constant wrong labels)
    if top_prob < 0.50:
        return "Uncertain"

    return LABEL_TO_NAME.get(top_label, str(top_label))


def predict_debug(image_bytes: bytes) -> dict:
    probs = _probs_from_bytes(image_bytes)
    top_label = max(probs, key=probs.get) if probs else None
    top_prob = probs[top_label] if probs and top_label else None

    top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]

    return {
        "build_marker": BUILD_MARKER,
        "active_variant": ACTIVE_VARIANT,
        "load_mode": LOAD_MODE,
        "labels": LABELS,
        "num_classes": NUM_CLASSES,
        "top_label": top_label,
        "top_prob": top_prob,
        "top3": top3,
    }
