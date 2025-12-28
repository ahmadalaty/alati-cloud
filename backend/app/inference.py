import json
import torch
import torchvision.transforms as T
from PIL import Image
import os

from .config import settings


# -------------------------
# Load labels
# -------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model_files")

with open(os.path.join(MODEL_DIR, "labels.json"), "r") as f:
    LABELS = json.load(f)


# -------------------------
# Model loading
# -------------------------
def load_model(model_variant: str = "resnet18"):
    if model_variant == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(weights=None)
        weights_path = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet50.pth")
    else:
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        weights_path = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet18.pth")

    num_classes = len(LABELS)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    state = torch.load(weights_path, map_location="cpu")

    # 🔑 CRITICAL FIX: handle state_dict vs full model
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        # model was saved directly
        model = state

    model.eval()
    return model


MODEL = load_model("resnet18")


# -------------------------
# Preprocessing
# -------------------------
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# -------------------------
# Inference
# -------------------------
def run_inference(image_path: str, eye: str):
    """
    image_path: path to JPG
    eye: 'left' or 'right' (kept for future eye-specific logic)
    """

    img = Image.open(image_path).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0)

    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.sigmoid(logits)[0]

    probs_dict = {
        LABELS[i]: float(probs[i].item())
        for i in range(len(LABELS))
    }

    top_label = max(probs_dict, key=probs_dict.get)
    top_prob = probs_dict[top_label]

    triage = "Refer" if top_prob >= 0.5 else "Routine"

    return {
        "model_variant": "resnet18",
        "eye": eye,
        "probs": probs_dict,
        "top_label": top_label,
        "top_prob": top_prob,
        "triage": triage,
    }
