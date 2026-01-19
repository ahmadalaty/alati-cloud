import json
import os
import hashlib
from io import BytesIO
from typing import Dict, Tuple

import torch
import torchvision.transforms as T
from PIL import Image

BUILD_MARKER = "INFERENCE_V6_STRICT_ONLY_AUDIT_2026_01_19"

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model_files")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# -------------------------
# Labels
# -------------------------
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels_raw = json.load(f)

# Expected labels.json format:
# [
#   {"code":"N","name":"normal"},
#   ...
# ]
if isinstance(labels_raw, list) and len(labels_raw) > 0 and isinstance(labels_raw[0], dict):
    LABELS = [x["code"] for x in labels_raw]
    CODE_TO_NAME = {x["code"]: x["name"] for x in labels_raw}
else:
    LABELS = list(labels_raw)
    CODE_TO_NAME = {
        "N": "normal",
        "D": "diabetic_retinopathy",
        "G": "glaucoma",
        "C": "cataract",
        "A": "amd",
        "H": "hypertensive_retinopathy",
        "M": "myopia",
        "O": "other",
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


def load_model(model_variant: str) -> Tuple[torch.nn.Module, str, str, str, int]:
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

    weights_size = os.path.getsize(weights_path)
    weights_sha = _file_sha256(weights_path)

    ckpt = torch.load(weights_path, map_location=DEVICE)

    # full model object saved
    if not isinstance(ckpt, dict):
        ckpt.eval()
        return ckpt, active, "full_model", weights_sha, weights_size

    state = _extract_state_dict(ckpt)
    if state is None:
        raise RuntimeError("Checkpoint format not understood")

    state = _clean_state_dict(state)

    # STRICT ONLY
    missing, unexpected = model.load_state_dict(state, strict=False)

    # We require strict behavior:
    # if missing/unexpected -> crash
    if missing or unexpected:
        raise RuntimeError(
            f"STRICT LOAD FAILED: missing={missing[:20]} unexpected={unexpected[:20]}"
        )

    model.eval()
    return model, active, "strict", weights_sha, weights_size


TRANSFORM = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

MODEL, ACTIVE_VARIANT, LOAD_MODE, WEIGHTS_SHA, WEIGHTS_SIZE = load_model(DEFAULT_VARIANT)


# Startup audit log (very important)
print(
    "[MODEL AUDIT]",
    "marker=", BUILD_MARKER,
    "variant=", ACTIVE_VARIANT,
    "load=", LOAD_MODE,
    "weights_sha=", WEIGHTS_SHA[:16],
    "weights_size=", WEIGHTS_SIZE,
    "labels=", LABELS,
    "fc_out=", MODEL.fc.out_features if hasattr(MODEL, "fc") else None,
)


def _probs_from_bytes(image_bytes: bytes) -> Dict[str, float]:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0)

    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.sigmoid(logits)[0].detach().cpu().tolist()

    out = {}
    for i in range(min(len(probs), len(LABELS))):
        out[str(LABELS[i])] = float(probs[i])
    return out


def translate_code(code: str) -> str:
    if not code:
        return "uncertain"
    name = CODE_TO_NAME.get(code, code)
    name = name.replace("_", " ").replace("-", " ").strip()
    name = " ".join(w.capitalize() for w in name.split())
    return name or "Uncertain"


def predict_raw(image_bytes: bytes) -> dict:
    probs = _probs_from_bytes(image_bytes)
    if not probs:
        return {"top_code": None, "top_prob": None, "top3": []}

    top_code = max(probs, key=probs.get)
    top_prob = probs[top_code]
    top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]

    return {
        "top_code": top_code,
        "top_prob": float(top_prob),
        "top3": [(k, float(v)) for k, v in top3],
    }


def predict_diagnosis(image_bytes: bytes) -> str:
    raw = predict_raw(image_bytes)
    top_code = raw.get("top_code")
    top_prob = raw.get("top_prob")

    if top_code is None or top_prob is None:
        return "Uncertain"

    # confidence guard
    if top_prob < 0.50:
        return "Uncertain"

    return translate_code(top_code)


def predict_debug(image_bytes: bytes) -> dict:
    raw = predict_raw(image_bytes)
    top_code = raw.get("top_code")
    return {
        "build_marker": BUILD_MARKER,
        "active_variant": ACTIVE_VARIANT,
        "load_mode": LOAD_MODE,
        "weights_sha": WEIGHTS_SHA,
        "weights_size": WEIGHTS_SIZE,
        "labels": LABELS,
        "num_classes": NUM_CLASSES,
        **raw,
        "translated": translate_code(top_code) if top_code else None,
    }
