import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    from model_single_task_learning import resnet50
except ImportError:
    print("[ERROR] model_single_task_learning.py를 찾을 수 없습니다.")
    sys.exit(1)


def build_transform(image_size: int, profile: str) -> transforms.Compose:
    """Transform profile compatible with make_vector_db.py and app.py."""
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


def _extract_state_dict(loaded_obj: object) -> Dict[str, torch.Tensor]:
    if isinstance(loaded_obj, dict) and "state_dict" in loaded_obj:
        state = loaded_obj["state_dict"]
    else:
        state = loaded_obj

    if not isinstance(state, dict):
        raise ValueError("Checkpoint format not supported. Expected state_dict or {'state_dict': ...}.")

    return state


def load_model_and_labels(
    model_path: str,
    label_map_path: str,
    dropout: float,
    device: torch.device,
):
    with open(label_map_path, encoding="utf-8") as f:
        label_to_index: Dict[str, int] = json.load(f)

    index_to_label = {v: k for k, v in label_to_index.items()}
    num_classes = len(label_to_index)

    model = resnet50(img_channels=3, num_classes=num_classes, dropout_p=dropout)
    loaded_obj = torch.load(model_path, map_location=device)
    state_dict = _extract_state_dict(loaded_obj)

    if any(k.startswith("backbone.") for k in state_dict.keys()):
        raise ValueError(
            "ResNet50Classifier(backbone/classifier) 체크포인트는 이 스크립트에서 지원하지 않습니다. "
            "model_single_task_learning.resnet50용 state_dict를 사용하세요."
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print(f"[WARN] state_dict mismatch | missing={len(missing)}, unexpected={len(unexpected)}")

    model.to(device)
    model.eval()
    return model, index_to_label


def split_label_to_style_gender(label: str) -> Tuple[str, str]:
    parts = label.rsplit("_", 1)
    if len(parts) != 2:
        return label, "unknown"
    style, gender = parts[0], parts[1].lower()
    if gender not in {"male", "female", "m", "w", "f"}:
        return style, "unknown"
    if gender == "m":
        gender = "male"
    elif gender in {"w", "f"}:
        gender = "female"
    return style, gender


def infer_single(
    image_path: str,
    model: torch.nn.Module,
    transform: transforms.Compose,
    index_to_label: Dict[int, str],
    device: torch.device,
    top_k: int = 5,
) -> List[Dict]:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    top_indices = probs.argsort(descending=True)[:top_k].tolist()
    results: List[Dict] = []
    for rank, idx in enumerate(top_indices, 1):
        label = index_to_label.get(idx, f"unknown_{idx}")
        style, gender = split_label_to_style_gender(label)
        results.append(
            {
                "rank": rank,
                "label": label,
                "style": style,
                "gender": gender,
                "probability": float(probs[idx]),
            }
        )
    return results


def infer_folder(
    image_dir: str,
    model: torch.nn.Module,
    transform: transforms.Compose,
    index_to_label: Dict[int, str],
    device: torch.device,
    top_k: int,
    output_csv: Optional[str],
) -> None:
    root = Path(image_dir)
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"[Infer] {len(paths)}장 추론 시작")

    rows: List[Dict[str, str]] = []
    for i, path in enumerate(paths, 1):
        try:
            results = infer_single(str(path), model, transform, index_to_label, device, top_k=top_k)
            top1 = results[0]
            topk_labels = "|".join([r["label"] for r in results])
            topk_probs = "|".join([f"{r['probability']:.4f}" for r in results])

            rows.append(
                {
                    "file": path.name,
                    "path": str(path),
                    "pred_label": top1["label"],
                    "pred_style": top1["style"],
                    "pred_gender": top1["gender"],
                    "prob": f"{top1['probability']:.4f}",
                    "topk_labels": topk_labels,
                    "topk_probs": topk_probs,
                }
            )
            print(
                f"\r  {i}/{len(paths)} | {path.name[:40]:<40} "
                f"-> {top1['label']} ({top1['probability']:.3f})",
                end="",
                flush=True,
            )
        except Exception as e:
            print(f"\n[WARN] 추론 실패: {path} | {e}")

    print()

    if output_csv:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file",
                    "path",
                    "pred_label",
                    "pred_style",
                    "pred_gender",
                    "prob",
                    "topk_labels",
                    "topk_probs",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Save] {output_csv}  ({len(rows)}행)")
    else:
        print("\n[결과 요약]")
        for r in rows[:20]:
            print(f"  {r['file']:<40} -> {r['pred_label']}  ({r['prob']})")
        if len(rows) > 20:
            print(f"  ... 외 {len(rows) - 20}건 (--output으로 전체 저장 가능)")


def main() -> None:
    parser = argparse.ArgumentParser(description="ResNet50 패션 스타일 추론")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="단일 이미지 경로")
    group.add_argument("--image-dir", help="폴더 전체 추론")

    parser.add_argument("--model", required=True, help="best_model_state.pth 경로")
    parser.add_argument("--label-map", required=True, help="label_to_index.json 경로")
    parser.add_argument("--output", default="", help="폴더 추론 결과 CSV 경로")
    parser.add_argument("--top-k", type=int, default=5, help="단일/폴더 Top-K 계산 수")
    parser.add_argument("--image-size", type=int, default=224)
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

    model, index_to_label = load_model_and_labels(args.model, args.label_map, args.dropout_p, device)
    transform = build_transform(args.image_size, args.transform_profile)

    if args.image:
        print(f"\n[이미지] {args.image}")
        results = infer_single(args.image, model, transform, index_to_label, device, args.top_k)
        print(f"\n{'Rank':<6} {'Label':<30} {'Style':<20} {'Gender':<10} {'Prob':>8}")
        print("-" * 76)
        for r in results:
            print(f"{r['rank']:<6} {r['label']:<30} {r['style']:<20} {r['gender']:<10} {r['probability']:>8.4f}")
    else:
        infer_folder(args.image_dir, model, transform, index_to_label, device, args.top_k, args.output or None)


if __name__ == "__main__":
    main()
