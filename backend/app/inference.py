import json
import os
from io import BytesIO
from typing import Dict, Tuple, List, Any

import torch
import torchvision.transforms as T
from PIL import Image

BUILD_MARKER = "INFERENCE_V5_LABELS_LIST_OF_DICTS_FIXED_2026_01_18"

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model_files")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

DEVICE = "cpu"
DEFAULT_VARIANT = os.getenv("MODEL_VARIANT", "resnet18").strip().lower()


# -------------------------
# Labels loader (YOUR FORMAT)
# -------------------------
def _load_labels_codes(path: str) -> List[str]:
    """
    Supports:
      - ["N","D","G"...]
      - [{"code":"N","name":"normal"}, ...]  <-- YOUR FILE
      - {"0":"N","1":"D"...}
    Returns list of codes in training order.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Case A: list[str]
    if isinstance(raw, list) and raw and isinstance(raw[0], str):
        return [str(x).strip().upper() for x in raw]

    # Case B: list[dict]  <-- YOUR FORMAT
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        codes = []
        for item in raw:
            code = str(item.get("code", "")).strip().upper()
            if not code:
                raise RuntimeError("labels.json has dict entry missing 'code'")
            codes.append(code)
        return codes

    # Case C: dict like {"0":"N"...}
    if isinstance(raw, dict):
        # try numeric index keys first
        try:
            return [str(raw[str(i)]).strip().upper() for i in range(len(raw))]
        except Exception:
            return [str(v).strip().upper() for v in raw.values()]

    raise RuntimeError("labels.json format not understood")


def _load_label_names(path: str) -> Dict[str, str]:
    """
    Returns mapping code -> name if available.
    For your labels.json list-of-dicts:
      {"N":"normal", "D":"diabetic_retinopathy", ...}
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        m = {}
        for item in raw:
            code = str(item.get("code", "")).strip().upper()
            name = str(item.get("name", "")).strip().lower()
            if code:
                m[code] = name
        return m
    return {}


LABELS: List[str] = _load_labels_codes(LABELS_PATH)
LABEL_NAMES: Dict[str, str] = _load_label_names(LABELS_PATH)

# fallback if names missing
DEFAULT_LABEL_TO_NAME = {
    "N": "normal",
    "D": "diabetic retinopathy",
    "G": "glaucoma",
    "C": "cataract",
    "A": "amd",
    "H": "hypertensive retinopathy",
    "M": "myopia",
    "O": "other",
}

LABEL_TO_NAME = {}
for c in LABELS:
    nm = LABEL_NAMES.get(c)
    if nm:
        LABEL_TO_NAME[c] = nm.replace("_", " ").strip().lower()
    else:
        LABEL_TO_NAME[c] = DEFAULT_LABEL_TO_NAME.get(c, c.lower())

NUM_CLASSES = len(LABELS)


# -------------------------
# Model loading helpers
# -------------------------
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


def _extract_state_dict(ckpt: Any):
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
        return ckpt
    return None


def load_model(model_variant: str) -> Tuple[torch.nn.Module, str, str, str]:
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

    # classifier head MUST match NUM_CLASSES
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

    if not os.path.exists(weights_path):
        raise RuntimeError(f"Model weights not found: {weights_path}")

    ckpt = torch.load(weights_path, map_location=DEVICE)

    # full model object
    if not isinstance(ckpt, dict):
        ckpt.eval()
        return ckpt, active, "full_model", weights_path

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
    return model, active, load_mode, weights_path


TRANSFORM = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

MODEL, ACTIVE_VARIANT, LOAD_MODE, WEIGHTS_PATH = load_model(DEFAULT_VARIANT)


# -------------------------
# Inference
# -------------------------
def _probs_from_bytes(image_bytes: bytes) -> Dict[str, float]:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0)

    with torch.no_grad():
        logits = MODEL(x)

        # Multi-label -> sigmoid
        probs = torch.sigmoid(logits)[0].detach().cpu().tolist()

    out: Dict[str, float] = {}
    for i in range(min(len(probs), len(LABELS))):
        out[LABELS[i]] = float(probs[i])
    return out


def predict_diagnosis(image_bytes: bytes) -> str:
    """
    Returns diagnosis STRING ONLY.
    """
    probs = _probs_from_bytes(image_bytes)
    if not probs:
        return "Uncertain"

    top_label = max(probs, key=probs.get)
    top_prob = probs[top_label]

    # safety threshold
    if top_prob < 0.50:
        return "Uncertain"

    diag = LABEL_TO_NAME.get(top_label, str(top_label))
    # polish for UI
    diag = diag.replace("_", " ").strip()
    diag = " ".join(w.capitalize() for w in diag.split())
    return diag


def predict_debug(image_bytes: bytes) -> dict:
    probs = _probs_from_bytes(image_bytes)
    if probs:
        top_label = max(probs, key=probs.get)
        top_prob = probs[top_label]
    else:
        top_label, top_prob = None, None

    top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]

    # important: verify head not random
    try:
        fc_mean = float(MODEL.fc.weight.mean().item())
        fc_std = float(MODEL.fc.weight.std().item())
    except Exception:
        fc_mean = None
        fc_std = None

    return {
        "build_marker": BUILD_MARKER,
        "active_variant": ACTIVE_VARIANT,
        "load_mode": LOAD_MODE,
        "weights_path": WEIGHTS_PATH,
        "num_classes": NUM_CLASSES,
        "labels_codes_order": LABELS,  # critical
        "label_to_name": LABEL_TO_NAME,
        "top_label": top_label,
        "top_prob": top_prob,
        "top3": top3,
        "fc_weight_mean": fc_mean,
        "fc_weight_std": fc_std,
    }
