import json
import os
from io import BytesIO
from typing import Dict, Tuple, List

import torch
import torchvision.transforms as T
from PIL import Image

BUILD_MARKER = "INFERENCE_V4_FIX_LABELS_LIST_OF_DICTS_2026_01_18"

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model_files")

LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels_raw = json.load(f)

# ✅ IMPORTANT: your labels.json is a LIST of dicts: [{"code":"N","name":"normal"}, ...]
# We normalize it to:
#   LABEL_CODES = ["N","D","G","C","A","H","M","O"]
#   CODE_TO_NAME = {"N":"normal", "D":"diabetic retinopathy", ...}

LABEL_CODES: List[str] = []
CODE_TO_NAME: Dict[str, str] = {}

if isinstance(labels_raw, list) and labels_raw and isinstance(labels_raw[0], dict):
    for item in labels_raw:
        code = str(item.get("code", "")).strip().upper()
        name = str(item.get("name", "")).strip().lower()

        if not code:
            continue

        # polish name
        name = name.replace("_", " ").replace("-", " ").strip()
        if not name:
            name = code.lower()

        LABEL_CODES.append(code)
        CODE_TO_NAME[code] = name

elif isinstance(labels_raw, list):
    # fallback: ["N","D","G"...]
    for x in labels_raw:
        code = str(x).strip().upper()
        LABEL_CODES.append(code)
        CODE_TO_NAME[code] = code.lower()

elif isinstance(labels_raw, dict):
    # fallback: {"0":"N","1":"D"...}
    try:
        for i in range(len(labels_raw)):
            code = str(labels_raw[str(i)]).strip().upper()
            LABEL_CODES.append(code)
            CODE_TO_NAME[code] = code.lower()
    except Exception:
        for _, v in labels_raw.items():
            code = str(v).strip().upper()
            LABEL_CODES.append(code)
            CODE_TO_NAME[code] = code.lower()
else:
    raise RuntimeError("labels.json format not understood")

NUM_CLASSES = len(LABEL_CODES)
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
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

MODEL, ACTIVE_VARIANT, LOAD_MODE = load_model(DEFAULT_VARIANT)


def _probs_from_bytes(image_bytes: bytes) -> Dict[str, float]:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0)

    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.sigmoid(logits)[0].detach().cpu().tolist()

    out: Dict[str, float] = {}
    for i in range(min(len(probs), len(LABEL_CODES))):
        out[LABEL_CODES[i]] = float(probs[i])

    return out


def _title_case(name: str) -> str:
    name = name.replace("_", " ").replace("-", " ").strip()
    return " ".join(w.capitalize() for w in name.split())


def predict_diagnosis(image_bytes: bytes) -> str:
    """
    ✅ Returns STRING ONLY (never dict)
    """
    probs = _probs_from_bytes(image_bytes)
    if not probs:
        return "Uncertain"

    top_code = max(probs, key=probs.get)
    top_prob = probs[top_code]

    # if model is not confident
    if top_prob < 0.50:
        return "Uncertain"

    raw_name = CODE_TO_NAME.get(top_code, top_code)
    return _title_case(raw_name)


def predict_debug(image_bytes: bytes) -> dict:
    probs = _probs_from_bytes(image_bytes)
    top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]

    return {
        "build_marker": BUILD_MARKER,
        "variant": ACTIVE_VARIANT,
        "load_mode": LOAD_MODE,
        "label_codes": LABEL_CODES,
        "code_to_name": CODE_TO_NAME,
        "top3": top3,
    }
