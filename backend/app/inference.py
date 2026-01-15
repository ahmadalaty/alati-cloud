import json
import os
from io import BytesIO
from typing import List, Any

import torch
import torchvision.transforms as T
from PIL import Image

# Marker to confirm correct file is running
BUILD_MARKER = "BACKEND_INFERENCE_DIAG_ONLY_v1_2026_01_15"

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model_files")


def _normalize_labels(raw: Any) -> List[str]:
    """
    Accepts labels.json in many shapes and returns a clean list[str] indexed by class id.
    Supported:
      - ["N","D","G",...]
      - {"0":"N","1":"D",...}
      - [{"label":"N"}, {"label":"D"}] or [{"name":"N"}, ...] or [{"0":"N"}, ...]
    """
    if raw is None:
        return []

    # dict: {"0":"N", ...}
    if isinstance(raw, dict):
        items = []
        for k, v in raw.items():
            try:
                idx = int(k)
            except Exception:
                continue
            # v can be str or dict
            if isinstance(v, str):
                label = v
            elif isinstance(v, dict):
                label = v.get("label") or v.get("name") or next(iter(v.values()), "")
            else:
                label = str(v)
            items.append((idx, str(label)))
        items.sort(key=lambda x: x[0])
        return [lbl for _, lbl in items if lbl]

    # list: ["N", ...] or [{"label":"N"}, ...]
    if isinstance(raw, list):
        out = []
        for x in raw:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, dict):
                label = x.get("label") or x.get("name")
                if not label and len(x) == 1:
                    label = str(next(iter(x.values())))
                out.append(str(label) if label else "")
            else:
                out.append(str(x))
        return [x for x in out if x]

    return []


# Load labels
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    _RAW_LABELS = json.load(f)

LABELS = _normalize_labels(_RAW_LABELS)
NUM_CLASSES = len(LABELS)

# Model choice
ACTIVE_VARIANT = os.getenv("MODEL_VARIANT", "resnet18").strip().lower() or "resnet18"


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
        if isinstance(ckpt.get("state_dict"), dict):
            return ckpt["state_dict"]
        if isinstance(ckpt.get("model_state_dict"), dict):
            return ckpt["model_state_dict"]
        return ckpt
    return None


def load_model(model_variant: str):
    model_variant = (model_variant or "resnet18").strip().lower()
    if model_variant == "resnet50":
        from torchvision.models import resnet50

        model = resnet50(weights=None)
        weights_path = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet50.pth")
        model_variant = "resnet50"
    else:
        from torchvision.models import resnet18

        model = resnet18(weights=None)
        weights_path = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet18.pth")
        model_variant = "resnet18"

    if NUM_CLASSES <= 0:
        raise RuntimeError("labels.json produced 0 classes after normalization")

    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

    ckpt = torch.load(weights_path, map_location="cpu")

    # full model saved
    if not isinstance(ckpt, dict):
        ckpt.eval()
        return ckpt, model_variant

    state = _extract_state_dict(ckpt)
    if state is None:
        raise RuntimeError("Checkpoint format not understood (no state_dict found)")

    state = _clean_state_dict(state)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=False)

    model.eval()
    return model, model_variant


MODEL, ACTIVE_VARIANT = load_model(ACTIVE_VARIANT)

TRANSFORM = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict_diagnosis(image_bytes: bytes) -> str:
    """
    Returns diagnosis ONLY as a string label (e.g., "N", "D", ...).
    Never returns dict/probs.
    """
    if not image_bytes:
        return "Unknown"

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0)

    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.sigmoid(logits)[0]

    # Safe top class
    best_i = int(torch.argmax(probs).item())
    if best_i < 0 or best_i >= len(LABELS):
        return "Unknown"

    label = LABELS[best_i]
    return str(label or "Unknown")
