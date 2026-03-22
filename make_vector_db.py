import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import faiss  # type: ignore
except ImportError:
    print("[ERROR] faiss is not installed.")
    print("  GPU: pip install faiss-gpu")
    print("  CPU: pip install faiss-cpu")
    sys.exit(1)

try:
    from model_single_task_learning import resnet50
except ImportError:
    print("[ERROR] model_single_task_learning.py is not found.")
    sys.exit(1)


VALID_EXTS = {".jpg", ".jpeg", ".png"}


def normalize_gender(token: Optional[str]) -> Optional[str]:
    if token is None:
        return None
    t = str(token).strip().lower()
    if t in {"male", "m", "man", "men"}:
        return "male"
    if t in {"female", "f", "w", "woman", "women"}:
        return "female"
    return None


def parse_style_gender(file_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse style/gender from file name.

    Priority:
    1) Legacy pattern: parts[3] = style, last token = W/M
    2) Fallback pattern: ..._<style>_<male/female>
    """
    stem = Path(file_name).stem
    parts = stem.split("_")

    style: Optional[str] = None
    gender: Optional[str] = None

    if len(parts) >= 5 and parts[3].strip():
        style = parts[3].strip().lower()

    if len(parts) >= 1:
        gender = normalize_gender(parts[-1])

    if style is None and gender is not None and len(parts) >= 2 and parts[-2].strip():
        style = parts[-2].strip().lower()

    return style, gender


def build_transform(image_size: int, profile: str) -> transforms.Compose:
    profile = profile.lower().strip()
    if profile == "team":
        resize = (2 * image_size, image_size)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif profile == "square":
        resize = (image_size, image_size)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif profile == "legacy":
        resize = (int(round(image_size * 4 / 3)), image_size)
        mean, std = (0.5498, 0.5226, 0.5052), (0.2600, 0.2582, 0.2620)
    else:
        raise ValueError("transform profile must be one of: team, square, legacy")

    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def extract_state_dict(loaded_obj: object) -> Dict[str, torch.Tensor]:
    if isinstance(loaded_obj, dict) and "state_dict" in loaded_obj:
        state = loaded_obj["state_dict"]
    else:
        state = loaded_obj

    if not isinstance(state, dict):
        raise ValueError("Checkpoint format not supported. Expected state_dict or {'state_dict': ...}.")

    return state


def infer_num_classes_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> int:
    if "fc.weight" in state_dict and hasattr(state_dict["fc.weight"], "shape"):
        return int(state_dict["fc.weight"].shape[0])
    raise ValueError("Cannot infer num_classes from checkpoint. Missing key: fc.weight")


def load_label_map(label_map_path: str, num_classes: int) -> Dict[str, int]:
    if label_map_path and Path(label_map_path).exists():
        with open(label_map_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        label_to_index = {str(k): int(v) for k, v in raw.items()}
        if len(label_to_index) != num_classes:
            print(
                f"[WARN] label map class count mismatch: {len(label_to_index)} vs checkpoint {num_classes}. "
                "Checkpoint class count will be used for model build."
            )
        return label_to_index

    print("[WARN] label map is not provided. Creating a fallback label map.")
    return {f"class_{i}": i for i in range(num_classes)}


class CatalogDataset(Dataset):
    def __init__(self, image_dir: str, transform: transforms.Compose, image_size: int = 224):
        self.transform = transform
        self.image_size = image_size

        root = Path(image_dir)
        if not root.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        all_paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS)

        self.samples: List[Dict[str, str]] = []
        skipped = 0
        for path in all_paths:
            style, gender = parse_style_gender(path.name)
            if style is None:
                skipped += 1
                continue
            self.samples.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "style": style,
                    "gender": gender or "unknown",
                }
            )

        print(f"[Dataset] valid images={len(self.samples)} | skipped={skipped}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.samples[idx]
        try:
            image = Image.open(item["path"]).convert("RGB")
        except Exception as e:
            print(f"\n[WARN] image load failed, using black image: {item['path']} | {e}")
            h = int(round(self.image_size * (4 / 3)))
            image = Image.new("RGB", (self.image_size, h), color=0)
        return self.transform(image), idx


def load_model(
    model_path: str,
    dropout: float,
    device: torch.device,
) -> Tuple[torch.nn.Module, int]:
    loaded_obj = torch.load(model_path, map_location=device)
    state_dict = extract_state_dict(loaded_obj)

    if any(k.startswith("backbone.") for k in state_dict.keys()):
        raise ValueError(
            "ResNet50Classifier(backbone/classifier) checkpoints are not supported in this script. "
            "Use model_single_task_learning.resnet50 state_dict."
        )

    num_classes = infer_num_classes_from_state_dict(state_dict)
    model = resnet50(img_channels=3, num_classes=num_classes, dropout_p=dropout)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print(f"[WARN] state_dict mismatch | missing={len(missing)}, unexpected={len(unexpected)}")

    model.to(device)
    model.eval()
    print(f"[Model] loaded: {model_path}")
    print(f"[Model] num_classes={num_classes} | feature_dim={model.fc.in_features}")
    return model, num_classes


def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    total: int,
    feature_dim: int,
) -> np.ndarray:
    all_vecs = np.zeros((total, feature_dim), dtype=np.float32)
    processed = 0
    t0 = time.time()

    for images, indices in loader:
        images = images.to(device)
        with torch.inference_mode():
            vecs = model.extract_feature_vector(images, normalize=True)
        vecs_np = vecs.cpu().numpy()

        for i, orig_idx in enumerate(indices.tolist()):
            all_vecs[orig_idx] = vecs_np[i]

        processed += len(indices)
        elapsed = time.time() - t0
        speed = processed / max(elapsed, 1e-6)
        eta = (total - processed) / max(speed, 1e-6)
        print(f"\r[Extract] {processed}/{total} | {speed:.1f} img/s | ETA {eta:.0f}s   ", end="", flush=True)

    print()
    print(f"[Extract] done: {processed} images | {time.time() - t0:.1f}s")
    return all_vecs


def save_all(
    output_dir: Path,
    samples: List[Dict[str, str]],
    embeddings: np.ndarray,
    label_to_index: Dict[str, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(feature_dim)
    index.add(embeddings)
    faiss.write_index(index, str(output_dir / "faiss.index"))
    print(f"[Save] faiss.index ({index.ntotal} vectors)")

    np.save(str(output_dir / "catalog_features.npy"), embeddings)
    print(f"[Save] catalog_features.npy shape={embeddings.shape}")

    csv_path = output_dir / "catalog_metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "style", "gender", "path"])
        writer.writeheader()
        for s in samples:
            writer.writerow(
                {
                    "image_name": s["name"],
                    "style": s["style"],
                    "gender": s["gender"],
                    "path": s["path"],
                }
            )
    print(f"[Save] catalog_metadata.csv ({len(samples)} rows)")

    lbl_path = output_dir / "label_to_index.json"
    with open(lbl_path, "w", encoding="utf-8") as f:
        json.dump(label_to_index, f, ensure_ascii=False, indent=2)
    print(f"[Save] label_to_index.json ({len(label_to_index)} classes)")

    print("\n[Check] top-5 neighbors for index 0")
    scores, indices = index.search(embeddings[0:1], 5)
    for rank, (s, i) in enumerate(zip(scores[0], indices[0]), 1):
        m = samples[i]
        print(f"  {rank}: [{m['style']}/{m['gender']}] {m['name']}  score={s:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS DB artifacts compatible with app.py")
    parser.add_argument("--image-dir", required=True, help="catalog image directory")
    parser.add_argument("--model-path", required=True, help="checkpoint path (best_model_state.pth)")
    parser.add_argument("--label-map", default="", help="label_to_index.json path (optional)")
    parser.add_argument("--output-dir", default="faiss_db", help="output directory")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (Windows: 0 recommended)")
    parser.add_argument("--dropout-p", type=float, default=0.2)
    parser.add_argument(
        "--transform-profile",
        choices=["team", "square", "legacy"],
        default="team",
        help="team: 2:1 resize + ImageNet norm / square: 1:1 + ImageNet norm / legacy: custom norm",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Config] device={device} | transform={args.transform_profile}")
    print(f"[Config] image_dir={args.image_dir}")
    print(f"[Config] output_dir={args.output_dir}")

    model, num_classes = load_model(args.model_path, args.dropout_p, device)
    label_to_index = load_label_map(args.label_map, num_classes)
    feature_dim = int(model.fc.in_features)

    transform = build_transform(args.image_size, args.transform_profile)
    dataset = CatalogDataset(args.image_dir, transform, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"\n[STEP] extract embeddings ({len(dataset)} images)")
    embeddings = extract_embeddings(model, loader, device, len(dataset), feature_dim)

    print("\n[STEP] save artifacts")
    save_all(Path(args.output_dir), dataset.samples, embeddings, label_to_index)

    print(f"\n[DONE] {Path(args.output_dir).resolve()}")
    print("\n[Use in app.py]")
    print(f"  Metadata CSV      : {args.output_dir}/catalog_metadata.csv")
    print(f"  Embeddings        : {args.output_dir}/catalog_features.npy")
    print(f"  FAISS Index       : {args.output_dir}/faiss.index")
    print(f"  Model checkpoint  : {args.model_path}")
    if args.label_map:
        print(f"  Label map json    : {args.label_map}")
    else:
        print(f"  Label map json    : {args.output_dir}/label_to_index.json (auto-generated)")
    print(f"  Transform profile : {args.transform_profile}")


if __name__ == "__main__":
    main()
