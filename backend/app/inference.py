import os
import json
from io import BytesIO
from PIL import Image

import torch
import torchvision.transforms as T

from .config import settings

BUILD_MARKER = "INFERENCE_BYTES_ONLY_v1_2026_01_11"

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model_files")

LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    LABELS = json.load(f)

# If your labels.json is ["N","D","G",...], we map to readable names:
ODIR8_NAMES = {
    "N": "Normal",
    "D": "Diabetic Retinopathy",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "Age-related Macular Degeneration",
    "H": "Hypertension",
    "M": "Myopia",
    "O": "Other",
}


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


def load_model():
    variant = (settings.MODEL_VARIANT or "resnet18").strip().lower()
    if variant == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(weights=None)
        weights_path = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet50.pth")
        active = "resnet50"
    else:
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        weights_path = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet18.pth")
        active = "resnet18"

    num_classes = len(LABELS)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    ckpt = torch.load(weights_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        ckpt.eval()
        return ckpt, active

    state = _extract_state_dict(ckpt)
    if state is None:
        raise RuntimeError("Checkpoint format not understood")

    state = _clean_state_dict(state)
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        model.load_state_dict(state, strict=False)

    model.eval()
    return model, active


MODEL, ACTIVE_VARIANT = load_model()

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def predict_diagnosis(image_bytes: bytes) -> str:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0)

    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.sigmoid(logits)[0]

    # pick top label
    top_i = int(torch.argmax(probs).item())
    raw_label = LABELS[top_i]

    # If raw label is a code like "N", map it. Otherwise return as-is.
    return ODIR8_NAMES.get(raw_label, str(raw_label))
