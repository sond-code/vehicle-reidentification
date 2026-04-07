import os
import math
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any

from flask import Flask, jsonify, render_template, request, send_from_directory, abort
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_ROOT = r"C:\Users\ml\Desktop\abhyudaya12\veri-vehicle-re-identification-dataset\versions\1\VeRi"
FEATURES_DIR = os.path.join(APP_ROOT, "features")
os.makedirs(FEATURES_DIR, exist_ok=True)

app = Flask(__name__)
app.config["DATASET_ROOT"] = os.environ.get("VERI_DATASET_ROOT", DEFAULT_DATASET_ROOT)

IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES_DIR = "features"
JOBS_FILE = os.path.join(FEATURES_DIR, "jobs.json")
os.makedirs(FEATURES_DIR, exist_ok=True)

if not os.path.exists(JOBS_FILE):
    with open(JOBS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

MODEL_BUILDERS = {
    "resnet50": lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
    "resnet18": lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
    "mobilenet_v2": lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
    "mobilenet_v3_small": lambda: models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT),
}

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_model_cache: Dict[str, nn.Module] = {}
_jobs: List[Dict[str, Any]] = []


def dataset_root() -> str:
    return app.config["DATASET_ROOT"]


def split_dirs() -> Dict[str, str]:
    root = dataset_root()
    return {
        "train": os.path.join(root, "image_train"),
        "query": os.path.join(root, "image_query"),
        "test": os.path.join(root, "image_test"),
    }


def list_images(split: str) -> List[str]:
    folder = split_dirs().get(split)
    if not folder or not os.path.isdir(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])


def parse_filename(filename: str) -> Dict[str, str]:
    parts = filename.split("_")
    vehicle_id = parts[0] if len(parts) > 0 else "unknown"
    camera_id = parts[1] if len(parts) > 1 else "unknown"
    return {"vehicle_id": vehicle_id, "camera_id": camera_id}


def get_model(name: str) -> nn.Module:
    if name in _model_cache:
        return _model_cache[name]
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported model: {name}")

    model = MODEL_BUILDERS[name]()
    if name.startswith("resnet"):
        backbone = nn.Sequential(*list(model.children())[:-1])
    elif name.startswith("mobilenet_v2"):
        backbone = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)))
    elif name.startswith("mobilenet_v3"):
        backbone = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)))
    else:
        raise ValueError(f"Unsupported model: {name}")

    backbone.eval().to(DEVICE)
    _model_cache[name] = backbone
    return backbone


def feature_dim(name: str) -> int:
    if name == "resnet50":
        return 2048
    if name == "resnet18":
        return 512
    if name == "mobilenet_v2":
        return 1280
    if name == "mobilenet_v3_small":
        return 576
    return 0


def extract_feature(image_path: str, model_name: str) -> torch.Tensor:
    model = get_model(model_name)
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = model(x)
        feat = torch.flatten(feat, 1)
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)
    return feat.cpu().squeeze(0)


def load_jobs():
    if not os.path.exists(JOBS_FILE):
        return []
    with open(JOBS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_jobs(jobs):
    with open(JOBS_FILE, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2)

def parse_veri_filename(filename):
    # Example: 0001_c001_00016450_0.jpg
    parts = filename.split("_")
    vehicle_id = parts[0]
    camera_id = parts[1]
    return vehicle_id, camera_id

def get_split_path(dataset_root, split_name):
    split_map = {
        "train": "image_train",
        "query": "image_query",
        "test": "image_test"
    }
    folder = split_map.get(split_name.lower())
    if not folder:
        raise ValueError("Invalid split name")
    return os.path.join(dataset_root, folder)

class VeRiDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        image_path = os.path.join(self.image_dir, filename)

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        vehicle_id, camera_id = parse_veri_filename(filename)

        return {
            "image": image,
            "filename": filename,
            "image_path": image_path,
            "vehicle_id": vehicle_id,
            "camera_id": camera_id
        }

def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    filenames = [item["filename"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    vehicle_ids = [item["vehicle_id"] for item in batch]
    camera_ids = [item["camera_id"] for item in batch]

    return {
        "images": images,
        "filenames": filenames,
        "image_paths": image_paths,
        "vehicle_ids": vehicle_ids,
        "camera_ids": camera_ids
    }

def get_feature_model(model_name):
    model_name = model_name.lower()

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        input_size = 224
        feature_dim = 2048

    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        input_size = 224
        feature_dim = 512

    elif model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        feature_extractor = model.features
        input_size = 224
        feature_dim = 1280

    elif model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        feature_extractor = model.features
        input_size = 224
        feature_dim = 576

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    feature_extractor.eval()
    return feature_extractor, input_size, feature_dim

def extract_features_for_split(dataset_root, split, model_name, batch_size=32):
    image_dir = get_split_path(dataset_root, split)
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Split folder not found: {image_dir}")

    model, input_size, feature_dim = get_feature_model(model_name)

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = VeRiDataset(image_dir=image_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_features = []
    all_image_paths = []
    all_filenames = []
    all_vehicle_ids = []
    all_camera_ids = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"]
            outputs = model(images)

            if model_name in ["resnet50", "resnet18"]:
                outputs = outputs.view(outputs.size(0), -1)
            else:
                outputs = torch.nn.functional.adaptive_avg_pool2d(outputs, (1, 1))
                outputs = outputs.view(outputs.size(0), -1)

            all_features.append(outputs.cpu())
            all_image_paths.extend(batch["image_paths"])
            all_filenames.extend(batch["filenames"])
            all_vehicle_ids.extend(batch["vehicle_ids"])
            all_camera_ids.extend(batch["camera_ids"])

    features_tensor = torch.cat(all_features, dim=0)

    output_name = f"{model_name}_{split}.pt"
    output_path = os.path.join(FEATURES_DIR, output_name)

    torch.save({
        "model": model_name,
        "split": split,
        "feature_dim": features_tensor.shape[1],
        "features": features_tensor,
        "image_paths": all_image_paths,
        "filenames": all_filenames,
        "vehicle_ids": all_vehicle_ids,
        "camera_ids": all_camera_ids
    }, output_path)

    return {
        "output_file": output_name,
        "output_path": output_path,
        "num_images": len(dataset),
        "feature_dim": int(features_tensor.shape[1])
    }

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/config", methods=["GET", "POST"])
def config_route():
    if request.method == "POST":
        data = request.get_json(force=True)
        new_root = data.get("dataset_root", "").strip()
        if not new_root or not os.path.isdir(new_root):
            return jsonify({"ok": False, "error": "Invalid dataset root"}), 400
        app.config["DATASET_ROOT"] = new_root
    return jsonify({"dataset_root": dataset_root()})


@app.route("/api/dashboard")
def dashboard():
    splits = {k: list_images(k) for k in ["train", "query", "test"]}
    all_files = []
    for split, names in splits.items():
        for n in names:
            meta = parse_filename(n)
            all_files.append({"split": split, **meta, "filename": n})

    total_images = len(all_files)
    vehicles = len(set(item["vehicle_id"] for item in all_files))
    cameras = len(set(item["camera_id"] for item in all_files))

    extracted_count = 0
    for file in os.listdir(FEATURES_DIR):
        if file.endswith(".pt"):
            extracted_count += 1

    percent = (extracted_count / total_images * 100.0) if total_images else 0.0

    return jsonify({
        "dataset_root": dataset_root(),
        "total_images": total_images,
        "train_count": len(splits["train"]),
        "query_count": len(splits["query"]),
        "test_count": len(splits["test"]),
        "total_vehicles": vehicles,
        "total_cameras": cameras,
        "features_extracted": extracted_count,
        "feature_percent": round(percent, 1),
        "jobs": _jobs[-10:][::-1],
    })


@app.route("/api/images")
def images_api():
    split = request.args.get("split", "train")
    vehicle_id = request.args.get("vehicle_id", "").strip()
    camera_id = request.args.get("camera_id", "").strip()
    has_features = request.args.get("has_features", "any")
    page = max(int(request.args.get("page", 1)), 1)
    page_size = max(min(int(request.args.get("page_size", 24)), 100), 1)

    files = []
    for filename in list_images(split):
        meta = parse_filename(filename)
        if vehicle_id and meta["vehicle_id"] != vehicle_id:
            continue
        if camera_id and meta["camera_id"] != camera_id:
            continue
        feat_key_prefix = f"{split}__{filename}__"
        feat_exists = any(name.startswith(feat_key_prefix) and name.endswith(".pt") for name in os.listdir(FEATURES_DIR))
        if has_features == "yes" and not feat_exists:
            continue
        if has_features == "no" and feat_exists:
            continue
        files.append({
            "filename": filename,
            "split": split,
            "vehicle_id": meta["vehicle_id"],
            "camera_id": meta["camera_id"],
            "has_features": feat_exists,
            "image_url": f"/api/image/{split}/{filename}",
        })

    total = len(files)
    total_pages = max(math.ceil(total / page_size), 1)
    start = (page - 1) * page_size
    end = start + page_size

    available_vehicles = sorted(set(parse_filename(f)["vehicle_id"] for f in list_images(split)))
    available_cameras = sorted(set(parse_filename(f)["camera_id"] for f in list_images(split)))

    return jsonify({
        "items": files[start:end],
        "total": total,
        "page": page,
        "total_pages": total_pages,
        "available_vehicles": available_vehicles,
        "available_cameras": available_cameras,
    })


@app.route("/api/image/<split>/<path:filename>")
def serve_image(split: str, filename: str):
    folder = split_dirs().get(split)
    if not folder or not os.path.isdir(folder):
        abort(404)
    return send_from_directory(folder, filename)


@app.route("/api/extract", methods=["POST"])
def extract_api():
    data = request.get_json(force=True)
    model_name = data.get("model", "resnet50")
    split = data.get("split", "train")
    custom_ids = [x.strip() for x in data.get("custom_ids", []) if str(x).strip()]
    batch_size = int(data.get("batch_size", 32))

    if model_name not in MODEL_BUILDERS:
        return jsonify({"ok": False, "error": "Unsupported model"}), 400
    if split not in {"train", "query", "test"}:
        return jsonify({"ok": False, "error": "Unsupported split"}), 400

    all_names = list_images(split)
    selected = []
    if custom_ids:
        allowed = set(custom_ids)
        for name in all_names:
            if parse_filename(name)["vehicle_id"] in allowed:
                selected.append(name)
    else:
        selected = all_names

    if not selected:
        return jsonify({"ok": False, "error": "No images matched the request"}), 400

    start_ts = datetime.now()
    processed = 0
    saved_files = []

    for filename in selected:
        img_path = os.path.join(split_dirs()[split], filename)
        feat = extract_feature(img_path, model_name)
        out_name = f"{split}__{filename}__{model_name}.pt"
        out_path = os.path.join(FEATURES_DIR, out_name)
        torch.save({
            "feature": feat,
            "filename": filename,
            "split": split,
            "vehicle_id": parse_filename(filename)["vehicle_id"],
            "camera_id": parse_filename(filename)["camera_id"],
            "model": model_name,
            "dim": int(feat.shape[0]),
        }, out_path)
        processed += 1
        saved_files.append(out_name)

    end_ts = datetime.now()
    job = {
        "id": len(_jobs) + 1,
        "model": model_name,
        "split": split,
        "count": processed,
        "batch_size": batch_size,
        "feature_dim": feature_dim(model_name),
        "started_at": start_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": end_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round((end_ts - start_ts).total_seconds(), 2),
    }
    _jobs.append(job)

    return jsonify({"ok": True, "job": job, "saved_files": saved_files[:10], "saved_count": len(saved_files)})


@app.route("/api/jobs")
def jobs_api():
    return jsonify({"jobs": _jobs[::-1]})



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
