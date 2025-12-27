import os
import json
from collections import OrderedDict
from typing import Dict, Any, Tuple

import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image


BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model_files")

MODEL_18_PATH = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet18.pth")
MODEL_50_PATH = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet50.pth")
LABELS_PATH   = os.path.join(MODEL_DIR, "labels.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if not isinstance(labels, list) or not labels:
        raise RuntimeError("labels.json must be a non-empty JSON list of class labels.")
    return labels


LABELS = load_labels()
NUM_CLASSES = len(LABELS)


TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def _extract_state_dict(loaded_obj) -> OrderedDict:
    """
    Accepts:
    - raw state_dict (OrderedDict)
    - checkpoint dict with keys like 'state_dict', 'model_state_dict', etc.
    Returns an OrderedDict suitable for model.load_state_dict().
    """
    # Case 1: already a state_dict
    if isinstance(loaded_obj, OrderedDict):
        return loaded_obj

    # Case 2: checkpoint dict
    if isinstance(loaded_obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in loaded_obj and isinstance(loaded_obj[key], (dict, OrderedDict)):
                sd = loaded_obj[key]
                return OrderedDict(sd) if not isinstance(sd, OrderedDict) else sd

        # Sometimes the dict IS the state_dict (plain dict of tensors)
        # Heuristic: contains tensor values and has typical layer keys
        if any(isinstance(v, torch.Tensor) for v in loaded_obj.values()):
            return OrderedDict(loaded_obj)

    raise RuntimeError(
        "Unsupported .pth format. Expected a state_dict (OrderedDict/dict) "
        "or a checkpoint dict containing state_dict."
    )


def _strip_module_prefix(state_dict: OrderedDict) -> OrderedDict:
    """
    If trained with DataParallel, keys are like 'module.conv1.weight'.
    Strip 'module.' so it loads on a normal model.
    """
    if not state_dict:
        return state_dict

    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        new_sd = OrderedDict()
        for k, v in state_dict.items():
            new_sd[k.replace("module.", "", 1)] = v
        return new_sd
    return state_dict


def _build_model(variant: str):
    """
    Build a ResNet with correct output layer size = NUM_CLASSES
    """
    v = (variant or "resnet18").strip().lower()

    if v == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, NUM_CLASSES)
        return model, "resnet50"

    # default: resnet18
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, NUM_CLASSES)
    return model, "resnet18"


def _load_model_weights(model: torch.nn.Module, weights_path: str):
    loaded = torch.load(weights_path, map_location="cpu")
    state_dict = _extract_state_dict(loaded)
    state_dict = _strip_module_prefix(state_dict)

    # Load with strict=False to tolerate minor key mismatches if any
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    model.to(DEVICE)
    model.eval()

    return missing, unexpected


# Cache the loaded model in memory
_CACHED_MODEL = None
_CACHED_VARIANT = None


def get_model(variant: str = "resnet18") -> Tuple[torch.nn.Module, str]:
    global _CACHED_MODEL, _CACHED_VARIANT

    v = (variant or "resnet18").strip().lower()
    if v not in ("resnet18", "resnet50"):
        v = "resnet18"

    if _CACHED_MODEL is not None and _CACHED_VARIANT == v:
        return _CACHED_MODEL, _CACHED_VARIANT

    model, chosen = _build_model(v)
    weights_path = MODEL_50_PATH if chosen == "resnet50" else MODEL_18_PATH

    if not os.path.exists(weights_path):
        raise RuntimeError(f"Model weights not found: {weights_path}")

    missing, unexpected = _load_model_weights(model, weights_path)

    # Optional: print to logs (helpful on Render)
    if missing:
        print(f"[inference] WARNING missing keys ({len(missing)}): {missing[:8]}{'...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"[inference] WARNING unexpected keys ({len(unexpected)}): {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")

    _CACHED_MODEL = model
    _CACHED_VARIANT = chosen
    return _CACHED_MODEL, _CACHED_VARIANT


def run_inference(image_path: str, eye: str = "left", model_variant: str = "resnet18") -> Dict[str, Any]:
    """
    eye parameter kept for future UI/logic (left/right).
    model_variant: 'resnet18' or 'resnet50'
    """
    model, chosen_variant = get_model(model_variant)

    img = Image.open(image_path).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0].detach().cpu()

    probs_dict = {LABELS[i]: float(probs[i].item()) for i in range(NUM_CLASSES)}
    top_label = max(probs_dict, key=probs_dict.get)
    top_prob = probs_dict[top_label]

    triage = "Refer" if top_prob >= 0.6 else "Observe"

    return {
        "model_variant": chosen_variant,
        "eye": (eye or "").lower(),
        "probs": probs_dict,
        "top_label": top_label,
        "top_prob": float(top_prob),
        "triage": triage,
    }
