import os
import json
import torch
from torchvision import models, transforms
from PIL import Image

LABELS_PATH = os.path.join(os.path.dirname(__file__), "model_files", "labels.json")

# Default preprocessing (stable). If your training used different preprocessing, we can match later.
_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_MODEL = None
_VARIANT = None
_LABEL_ITEMS = None

def _load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list) or len(items) != 8:
        raise RuntimeError("labels.json must be a list of 8 items in ODIR order: N,D,G,C,A,H,M,O")
    return items

def _build_model(variant: str, num_outputs: int):
    variant = (variant or "resnet18").lower().strip()
    if variant == "resnet50":
        m = models.resnet50(weights=None)
    else:
        m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, num_outputs)
    return m

def _load_weights(model, weights_path: str):
    """
    Supports both:
    - torch.save(model.state_dict(), path)
    - torch.save(model, path)
    Also handles 'state_dict' nesting and DataParallel 'module.' prefixes.
    """
    obj = torch.load(weights_path, map_location="cpu")

    # Full model saved
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        return obj

    # state_dict saved (possibly nested)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        state = obj["state_dict"]
    else:
        state = obj

    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model

def _get_model():
    global _MODEL, _VARIANT, _LABEL_ITEMS

    if _LABEL_ITEMS is None:
        _LABEL_ITEMS = _load_labels()

    variant = os.getenv("MODEL_VARIANT", "resnet18").lower().strip()
    weights_file = "alati_dualeye_model_resnet18.pth" if variant != "resnet50" else "alati_dualeye_model_resnet50.pth"
    weights_path = os.path.join(os.path.dirname(__file__), "model_files", weights_file)

    if _MODEL is None or _VARIANT != variant:
        base = _build_model(variant, num_outputs=len(_LABEL_ITEMS))
        _MODEL = _load_weights(base, weights_path)
        _VARIANT = variant

    return _MODEL, _VARIANT, _LABEL_ITEMS

def run_inference(image_path: str) -> dict:
    """
    ODIR style: multi-label -> sigmoid per class (NOT softmax).
    """
    model, variant, label_items = _get_model()

    img = Image.open(image_path).convert("RGB")
    x = _preprocess(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)                 # [1,8]
        probs = torch.sigmoid(logits)[0]  # [8]

    probs_list = probs.cpu().numpy().tolist()

    prob_map = {}
    for i, item in enumerate(label_items):
        prob_map[item["name"]] = round(float(probs_list[i]), 4)

    top_i = int(max(range(len(probs_list)), key=lambda i: probs_list[i]))
    top_item = label_items[top_i]
    top_prob = float(probs_list[top_i])

    threshold = float(os.getenv("TRIAGE_THRESHOLD", "0.6"))
    normal_idx = next((i for i, it in enumerate(label_items) if it.get("code") == "N"), None)

    refer = False
    for i, p in enumerate(probs_list):
        if normal_idx is not None and i == normal_idx:
            continue
        if float(p) >= threshold:
            refer = True
            break

    triage = "Refer" if refer else "Normal"

    return {
        "model_variant": variant,
        "probs": prob_map,
        "top_label": top_item["name"],
        "top_code": top_item.get("code"),
        "top_prob": round(top_prob, 4),
        "triage": triage,
    }
