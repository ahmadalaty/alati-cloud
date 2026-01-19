import os
import json
import hashlib
from io import BytesIO
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image, ImageOps

BUILD_MARKER = "INFERENCE_V6_DUALEYE_MATCH_TRAINING_MIRROR_SINGLE_2026_01_19"

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model_files")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")


# -----------------------------
# Labels
# labels.json format:
# [
#   {"code":"N","name":"normal"},
#   ...
# ]
# -----------------------------
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels_raw = json.load(f)

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
DEFAULT_VARIANT = os.getenv("MODEL_VARIANT", "resnet18").strip().lower()
DEVICE = "cpu"


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def translate_code(code: str) -> str:
    """
    Translate ODIR code to UI-friendly label.
    Returns Title Case string.
    """
    if not code:
        return "Uncertain"
    name = CODE_TO_NAME.get(code, code)
    name = str(name).replace("_", " ").replace("-", " ").strip()
    name = " ".join(w.capitalize() for w in name.split())
    return name or "Uncertain"


# -----------------------------
# DualEye Model (MATCH TRAINING)
# -----------------------------
class DualEyeModel(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int = 8):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "resnet50":
            self.backbone = models.resnet50(weights=None)
            out_dim = 2048
        else:
            self.backbone = models.resnet18(weights=None)
            out_dim = 512

        # remove classifier head from backbone
        self.backbone.fc = nn.Identity()

        # classifier exactly like your train.py
        self.classifier = nn.Sequential(
            nn.Linear(out_dim * 2, 256),
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


# -----------------------------
# Image transforms (cloud)
# Must match inference pipeline stable
# -----------------------------
TRANSFORM = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
    ]
)


def _preprocess(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0)
    return x


def _mirror_bytes(image_bytes: bytes) -> bytes:
    """
    Mirror flip to simulate the other eye.
    Keeps distribution closer to training (dual-eye).
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    mirrored = ImageOps.mirror(img)
    out = BytesIO()
    mirrored.save(out, format="JPEG", quality=95)
    return out.getvalue()


def load_model(model_variant: str) -> Tuple[nn.Module, str, str, str, int]:
    model_variant = (model_variant or "resnet18").strip().lower()

    if model_variant == "resnet50":
        weights_path = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet50.pth")
        active = "resnet50"
    else:
        weights_path = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet18.pth")
        active = "resnet18"

    if not os.path.exists(weights_path):
        raise RuntimeError(f"Model weights not found: {weights_path}")

    model = DualEyeModel(active, num_classes=NUM_CLASSES).to(DEVICE)

    ckpt = torch.load(weights_path, map_location=DEVICE)
    if not isinstance(ckpt, dict):
        # someone saved entire model object
        ckpt.eval()
        return ckpt, active, "full_model", _sha256_file(weights_path), os.path.getsize(weights_path)

    # standard: state_dict
    try:
        model.load_state_dict(ckpt, strict=True)
        load_mode = "strict"
    except Exception:
        # allow minor mismatch if keys have prefix
        cleaned = {}
        for k, v in ckpt.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            cleaned[nk] = v
        model.load_state_dict(cleaned, strict=True)
        load_mode = "strict_cleaned"

    model.eval()
    return model, active, load_mode, _sha256_file(weights_path), os.path.getsize(weights_path)


MODEL, ACTIVE_VARIANT, LOAD_MODE, WEIGHTS_SHA, WEIGHTS_SIZE = load_model(DEFAULT_VARIANT)


def predict_raw_dual(left_bytes: bytes, right_bytes: bytes) -> dict:
    """
    Dual-eye inference (true trained path).
    Returns: {top_code, top_prob, top3}
    """
    left_x = _preprocess(left_bytes).to(DEVICE)
    right_x = _preprocess(right_bytes).to(DEVICE)

    with torch.no_grad():
        out = MODEL(left_x, right_x)  # shape [1, 8]
        probs = out[0].detach().cpu().tolist()

    probs_map = {}
    for i in range(min(len(probs), len(LABELS))):
        probs_map[LABELS[i]] = float(probs[i])

    top_code = max(probs_map, key=probs_map.get)
    top_prob = probs_map[top_code]
    top3 = sorted(probs_map.items(), key=lambda kv: kv[1], reverse=True)[:3]

    return {
        "top_code": top_code,
        "top_prob": float(top_prob),
        "top3": [(k, float(v)) for k, v in top3],
    }


def predict_single(image_bytes: bytes, eye_mode: str) -> dict:
    """
    Single-eye inference using mirror flip to produce pseudo dual-eye.
    eye_mode: left | right
    """
    eye_mode = (eye_mode or "left").strip().lower()
    mirrored = _mirror_bytes(image_bytes)

    if eye_mode == "right":
        # right = original, left = mirrored
        left_bytes = mirrored
        right_bytes = image_bytes
    else:
        # left = original, right = mirrored
        left_bytes = image_bytes
        right_bytes = mirrored

    return predict_raw_dual(left_bytes, right_bytes)


def predict_diagnosis(image_bytes: bytes, eye_mode: str = "left") -> str:
    """
    Always returns polished diagnosis string (no dict).
    """
    raw = predict_single(image_bytes, eye_mode)
    top_code = raw.get("top_code")
    top_prob = raw.get("top_prob")

    if not top_code or top_prob is None:
        return "Uncertain"

    # confidence guard
    if float(top_prob) < 0.50:
        return "Uncertain"

    return translate_code(top_code)


def predict_debug(image_bytes: bytes, eye_mode: str = "left") -> dict:
    raw = predict_single(image_bytes, eye_mode)
    return {
        "build_marker": BUILD_MARKER,
        "active_variant": ACTIVE_VARIANT,
        "load_mode": LOAD_MODE,
        "weights_sha256": WEIGHTS_SHA,
        "weights_size": WEIGHTS_SIZE,
        "labels": LABELS,
        "num_classes": NUM_CLASSES,
        "eye_mode": eye_mode,
        **raw,
        "translated": translate_code(raw.get("top_code")),
    }
