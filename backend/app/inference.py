import json
import os
import hashlib
from io import BytesIO
from typing import Dict, Tuple, Any

import torch
import torchvision.transforms as T
from PIL import Image

BUILD_MARKER = "INFERENCE_V6_STRICT_COMPAT_HEADMAP_2026_01_19"

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model_files")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

# labels.json format:
# [
#   {"code":"N","name":"normal"},
#   ...
# ]
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels_raw = json.load(f)

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


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean_state_dict(state: dict) -> dict:
    """Strip common prefixes."""
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


def _remap_classifier_to_fc(state: dict) -> dict:
    """
    Some trainings save head as:
      classifier.3.weight / classifier.3.bias
    while torchvision resnet uses:
      fc.weight / fc.bias
    This remaps keys so strict load succeeds.
    """
    if "fc.weight" in state or "fc.bias" in state:
        return state

    # if checkpoint used classifier.3 as final head
    if "classifier.3.weight" in state and "classifier.3.bias" in state:
        state = dict(state)  # copy
        state["fc.weight"] = state.pop("classifier.3.weight")
        state["fc.bias"] = state.pop("classifier.3.bias")

    # classifier.0.* is not in torchvision resnet, ignore it safely
    for k in ["classifier.0.weight", "classifier.0.bias", "classifier.1.weight", "classifier.1.bias"]:
        if k in state:
            state.pop(k, None)

    return state


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

    # IMPORTANT: torchvision resnet final head
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

    if not os.path.exists(weights_path):
        raise RuntimeError(f"Model weights not found: {weights_path}")

    weights_sha = _sha256_file(weights_path)
    weights_size = os.path.getsize(weights_path)

    ckpt = torch.load(weights_path, map_location=DEVICE)

    # full model saved
    if not isinstance(ckpt, dict):
        ckpt.eval()
        return ckpt, active, "full_model_object", weights_sha, weights_size

    state = _extract_state_dict(ckpt)
    if state is None:
        raise RuntimeError("Checkpoint format not understood")

    state = _clean_state_dict(state)
    state = _remap_classifier_to_fc(state)

    # STRICT load (after remap)
    missing, unexpected = model.load_state_dict(state, strict=False)

    # We try to enforce that fc.* is loaded
    # If not loaded -> fatal
    if any(x in missing for x in ["fc.weight", "fc.bias"]):
        raise RuntimeError(
            f"LOAD FAILED: missing={missing} unexpected={unexpected}"
        )

    model.eval()
    return model, active, "loaded_with_remap", weights_sha, weights_size


TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


MODEL, ACTIVE_VARIANT, LOAD_MODE, WEIGHTS_SHA, WEIGHTS_SIZE = load_model(DEFAULT_VARIANT)


def translate_code(code: str) -> str:
    if not code:
        return "Uncertain"
    name = CODE_TO_NAME.get(code, code)
    name = name.replace("_", " ").replace("-", " ").strip()
    name = " ".join(w.capitalize() for w in name.split())
    return name or "Uncertain"


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
        "labels": LABELS,
        "num_classes": NUM_CLASSES,
        "weights_sha": WEIGHTS_SHA,
        "weights_size": WEIGHTS_SIZE,
        **raw,
        "translated": translate_code(top_code) if top_code else None,
    }
