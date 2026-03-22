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
    from model_single_task_learning_team import ResNet50Classifier # 여기는 본인 걸로 바꾸기
except ImportError:
    print("[ERROR] model_single_task_learning_team.py를 찾을 수 없습니다.")
    sys.exit(1)


def build_transform(image_size: int) -> transforms.Compose:
    """Transform compatible with team model training/inference path."""
    resize = (2 * image_size, image_size)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _extract_state_dict(loaded_obj: object) -> Dict[str, torch.Tensor]:
    state = loaded_obj
    if isinstance(loaded_obj, dict):
        for key in ["state_dict", "model_state_dict"]:
            if key in loaded_obj and isinstance(loaded_obj[key], dict):
                state = loaded_obj[key]
                break

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

    model = ResNet50Classifier(
        num_classes=num_classes,
        dropout=dropout,
        pretrained=False,
    )

    try:
        loaded_obj = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        loaded_obj = torch.load(model_path, map_location=device)
    state_dict = _extract_state_dict(loaded_obj)

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
    print(f"[Infer] {len(paths)}개 추론 시작")

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
        print(f"[Save] {output_csv}  ({len(rows)}개)")
    else:
        print("\n[결과 요약]")
        for r in rows[:20]:
            print(f"  {r['file']:<40} -> {r['pred_label']}  ({r['prob']})")
        if len(rows) > 20:
            print(f"  ... 외 {len(rows) - 20}건 (--output으로 전체 저장 가능)")


def main() -> None:
    parser = argparse.ArgumentParser(description="ResNet50 패션 스타일 추론 (team model 전용)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="단일 이미지 경로")
    group.add_argument("--image-dir", help="폴더 전체 추론")
    parser.add_argument("--model", required=True, help="best_model_state.pth 경로")
    parser.add_argument("--label-map", required=True, help="label_to_index.json 경로")
    parser.add_argument("--output", default="", help="폴더 추론 결과 CSV 경로")
    parser.add_argument("--top-k", type=int, default=5, help="단일/폴더 Top-K 계산 수")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dropout-p", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Config] device={device}")
    model, index_to_label = load_model_and_labels(args.model, args.label_map, args.dropout_p, device)
    transform = build_transform(args.image_size)

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
