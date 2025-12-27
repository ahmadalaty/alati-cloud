import json
import torch
import torchvision.transforms as T
from PIL import Image
import os

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model_files")

MODEL_18_PATH = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet18.pth")
MODEL_50_PATH = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet50.pth")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


LABELS = load_labels()


def load_model():
    model = torch.load(MODEL_18_PATH, map_location=DEVICE)
    model.eval()
    return model


MODEL = load_model()


TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def run_inference(image_path: str, eye: str = "left"):
    """
    eye parameter kept for future use (left/right handling).
    """

    img = Image.open(image_path).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.sigmoid(logits)[0]

    probs_dict = {
        LABELS[i]: float(probs[i].item())
        for i in range(len(LABELS))
    }

    top_label = max(probs_dict, key=probs_dict.get)
    top_prob = probs_dict[top_label]

    triage = "Refer" if top_prob >= 0.6 else "Observe"

    return {
        "model_variant": "resnet18",
        "probs": probs_dict,
        "top_label": top_label,
        "top_prob": top_prob,
        "triage": triage,
    }
