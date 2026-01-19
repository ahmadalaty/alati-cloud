import os
import json
from io import BytesIO
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image

BUILD_MARKER = "INFERENCE_DUALEYE_RESNET50_MATCH_TRAIN_2026_01_19"

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model_files")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

DEFAULT_VARIANT = os.getenv("MODEL_VARIANT", "resnet50").strip().lower()
DEVICE = "cpu"


# --------------------------
# Labels
# --------------------------
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels_raw = json.load(f)

# expected:
# [
#   {"code":"N","name":"normal"},
#   ...
# ]
if isinstance(labels_raw, list) and len(labels_raw) > 0 and isinstance(labels_raw[0], dict):
    LABELS: List[str] = [x["code"] for x in labels_raw]
    CODE_TO_NAME = {x["code"]: x["name"] for x in labels_raw}
else:
    # fallback
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


def translate_code(code: str) -> str:
    if not code:
        return "Uncertain"
    name = CODE_TO_NAME.get(code, code)
    name = name.replace("_", " ").replace("-", " ").strip()
    name = " ".join(w.capitalize() for w in name.split())
    return name or "Uncertain"


# --------------------------
# Model (MATCH TRAIN.PY)
# --------------------------
class DualEyeModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(2048 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, left, right):
        left_feat = self.backbone(left)
        right_feat = self.backbone(right)
        combined = torch.cat((left_feat, right_feat), dim=1)
        return self.classifier(combined)


TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])


def _load_image_tensor(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0)  # [1,3,224,224]
    return x


def _clean_state_dict(state: dict) -> dict:
    """
    training saved model.state_dict() where keys look like:
      backbone.xxx
      classifier.xxx
    We keep that as-is.
    But if it contains module., strip it.
    """
    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        cleaned[nk] = v
    return cleaned


def load_model() -> Tuple[nn.Module, str]:
    """
    Always load DualEyeModel ResNet50 (matches training).
    """
    weights_path = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet50.pth")
    if not os.path.exists(weights_path):
        raise RuntimeError(f"Model weights not found: {weights_path}")

    model = DualEyeModel(NUM_CLASSES).to(DEVICE)

    ckpt = torch.load(weights_path, map_location=DEVICE)
    if not isinstance(ckpt, dict):
        # full model object saved (rare)
        ckpt.eval()
        return ckpt, "full_model_object"

    state = _clean_state_dict(ckpt)

    # strict should work if architecture matches training
    missing, unexpected = model.load_state_dict(state, strict=False)

    model.eval()

    load_mode = "strict_ok"
    if missing or unexpected:
        load_mode = f"non_strict missing={len(missing)} unexpected={len(unexpected)}"

    return model, load_mode


MODEL, LOAD_MODE = load_model()
ACTIVE_VARIANT = "resnet50_dualeye"


# --------------------------
# Inference
# --------------------------
def _probs_from_bytes(left_bytes: bytes, right_bytes: bytes) -> Dict[str, float]:
    left_x = _load_image_tensor(left_bytes)
    right_x = _load_image_tensor(right_bytes)

    with torch.no_grad():
        probs = MODEL(left_x, right_x)[0].detach().cpu().tolist()

    out = {}
    for i in range(min(len(probs), len(LABELS))):
        out[str(LABELS[i])] = float(probs[i])
    return out


def predict_raw(left_bytes: bytes, right_bytes: bytes) -> dict:
    """
    Returns RAW AI output:
      {top_code, top_prob, top3}
    """
    probs = _probs_from_bytes(left_bytes, right_bytes)
    if not probs:
        return {"top_code": None, "top_prob": None, "top3": []}

    top_code = max(probs, key=probs.get)
    top_prob = float(probs[top_code])
    top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]

    return {
        "top_code": top_code,
        "top_prob": top_prob,
        "top3": [(k, float(v)) for k, v in top3],
    }


def predict_diagnosis(left_bytes: bytes, right_bytes: bytes) -> str:
    """
    Always returns clean UI diagnosis string only.
    """
    raw = predict_raw(left_bytes, right_bytes)
    top_code = raw.get("top_code")
    top_prob = raw.get("top_prob")

    if top_code is None or top_prob is None:
        return "Uncertain"

    # confidence guard
    if top_prob < 0.50:
        return "Uncertain"

    return translate_code(top_code)


def predict_debug(left_bytes: bytes, right_bytes: bytes) -> dict:
    raw = predict_raw(left_bytes, right_bytes)
    top_code = raw.get("top_code")

    return {
        "build_marker": BUILD_MARKER,
        "active_variant": ACTIVE_VARIANT,
        "load_mode": LOAD_MODE,
        "labels": LABELS,
        "num_classes": NUM_CLASSES,
        **raw,
        "translated": translate_code(top_code) if top_code else None,
    }
