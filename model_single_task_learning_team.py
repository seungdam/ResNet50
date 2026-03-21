import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

# -----------------------------------------------------------------------------
# 이 파일의 전체 목적
# 1) 파일명에서 라벨(스타일/성별)을 읽어 학습 데이터셋을 구성
# 2) 단일 ResNet50 모델로 스타일+성별 결합 클래스(예: casual_male) 분류 학습
# 3) 평가 지표(Accuracy, F1, Confusion Matrix) 저장
# 4) 추천 시스템에서 재사용할 수 있는 feature vector(임베딩) 추출 지원
# -----------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """
    실험 재현성을 높이기 위한 시드 고정 함수.
    같은 코드/데이터/환경에서 결과를 최대한 비슷하게 만들기 위해 사용합니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_image_metadata(file_name: str) -> Optional[Dict[str, str]]:
    """
    파일명 형식:
    {prefix}_{이미지ID}_{시대별}_{스타일별}_{W/M}.jpg

    마지막 토큰은 성별 토큰(W/M)으로 가정합니다.
    """
    suffix = Path(file_name).suffix.lower()
    if suffix not in {".jpg", ".jpeg"}:
        return None

    stem = Path(file_name).stem
    parts = stem.split("_")
    if len(parts) < 5:
        return None

    image_id = parts[1]
    style = parts[3]
    gender_token = parts[-1].upper()
    if gender_token not in {"W", "M"}:
        return None
    gender = "female" if gender_token == "W" else "male"

    if not image_id or not style:
        return None

    return {
        "image_id": image_id,
        "style": style,
        "gender_token": gender_token,
        "gender": gender,
    }


def collect_records(image_dir: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    이미지 폴더를 순회하며 학습에 필요한 메타데이터를 records로 만듭니다.
    반환:
    - records: 정상 파싱된 샘플 목록
    - invalid_files: 파일명 규칙이 맞지 않거나 라벨 파싱 불가능한 파일 목록
    """
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    records: List[Dict[str, str]] = []
    invalid_files: List[str] = []

    image_paths = sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}]
    )

    for path in image_paths:
        meta = parse_image_metadata(path.name)
        if meta is None:
            invalid_files.append(path.name)
            continue

        records.append(
            {
                "path": str(path),
                "style": meta["style"],
                "gender": meta["gender"],
                "label": f"{meta['style']}_{meta['gender']}",
                "image_id": meta["image_id"],
            }
        )

    return records, invalid_files


def save_label_distribution(records: List[Dict[str, str]], csv_path: Path) -> None:
    """
    라벨 분포를 CSV로 저장합니다.
    팀원이 데이터 불균형(어떤 스타일 사진이 적은지)을 빠르게 확인할 때 유용합니다.
    """
    df = pd.DataFrame(records)
    if df.empty:
        pd.DataFrame(columns=["gender", "style", "label", "count"]).to_csv(
            csv_path, index=False, encoding="utf-8-sig"
        )
        return

    dist_df = (
        df.groupby(["gender", "style", "label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["gender", "style"])
    )
    dist_df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def split_train_val_records(
    records: List[Dict[str, str]], val_ratio: float, seed: int
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Train/Validation 분할 함수.
    가능하면 stratify(라벨 비율 유지)를 사용해 train/val 분포를 비슷하게 맞춥니다.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if len(records) < 2:
        raise ValueError("Not enough images to split train/val.")

    # 결합 라벨(style+gender) 기준으로 stratify를 시도하면 train/val 분포가 더 안정적입니다.
    stratify_keys = [r["label"] for r in records]
    key_counts = Counter(stratify_keys)
    can_stratify = len(key_counts) > 1 and min(key_counts.values()) >= 2

    train_records, val_records = train_test_split(
        records,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify_keys if can_stratify else None,
    )
    return train_records, val_records


class FashionStyleDataset(Dataset):
    """
    PyTorch Dataset.
    한 샘플을 꺼낼 때 (이미지 텐서, 결합 클래스 인덱스)를 반환합니다.
    """

    def __init__(
        self,
        records: List[Dict[str, str]],
        label_to_index: Dict[str, int],
        transform: Optional[transforms.Compose] = None,
        image_size = 224
    ):
        self.transform = transform
        self.samples = [r for r in records if r["label"] in label_to_index]
        self.label_to_index = label_to_index
        self.image_size = image_size
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        item = self.samples[idx]
        # PIL로 이미지를 열고 RGB 3채널로 통일
        try:
            image = Image.open(item["path"]).convert("RGB")
        except Exception as e:
            print(f"[WARN] 이미지 로드 실패, 검은 이미지 대체: {item['path']} | {e}")
            image = Image.new("RGB", (self.image_size, 2 * self.image_size), color=0)
        if self.transform is not None:
            # augmentation / resize / normalize 적용
            image = self.transform(image)
        label_idx = self.label_to_index[item["label"]]
        return image, label_idx


class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.2, pretrained: bool = True):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = models.resnet50(weights=weights)

        in_features = base.fc.in_features
        base.fc = nn.Identity()

        self.backbone = base
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x, return_features: bool = False):
        feat = self.backbone(x)         # [B, 2048]
        logits = self.classifier(self.dropout(feat))
        if return_features:
            return logits, feat
        return logits

    def extract_feature_vector(self, x, normalize: bool = True):
        feat = self.backbone(x)
        if normalize:
            feat = F.normalize(feat, p=2, dim=1)
        return feat


def create_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    학습/검증 전처리 파이프라인 구성.
    - train: augmentation 포함
    - val: resize + normalize 중심
    """
    train_transform = transforms.Compose([
    transforms.Resize((2 * image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]     
    ),
    ])

    val_transform = transforms.Compose([
    transforms.Resize((2 * image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225],
    ),
    ])
    
    
    return train_transform, val_transform

def build_class_weights(
    train_records: List[Dict[str, str]],
    label_to_index: Dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    """
    클래스 불균형 완화를 위한 손실 가중치 계산.
    샘플이 적은 클래스일수록 더 큰 가중치가 부여됩니다.
    """
    class_counts = np.zeros(len(label_to_index), dtype=np.float32)
    for rec in train_records:
        class_counts[label_to_index[rec["label"]]] += 1.0

    # 샘플 수가 적은 클래스의 손실을 조금 더 크게 주기 위한 역빈도 가중치
    weights = class_counts.sum() / (len(class_counts) * np.clip(class_counts, 1.0, None))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    log_prefix: str = "",
    log_interval: int = 0,
    data_wait_warn_sec: float = 10.0,
) -> Tuple[float, float, List[int], List[int]]:
    """
    지정한 데이터셋(loader)에서 loss/accuracy/prediction 목록을 계산.
    반환된 labels/preds는 confusion matrix, F1 계산에 사용됩니다.
    """
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    correct = 0
    total = 0

    eval_prev_end = time.time()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            batch_fetch_done = time.time()
            data_wait = batch_fetch_done - eval_prev_end
            step_start = time.time()
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == len(loader)):
                step_time = time.time() - step_start
                warn_text = ""
                if data_wait > data_wait_warn_sec:
                    warn_text = f" | [WARN] data_wait>{data_wait_warn_sec:.1f}s"
                print(
                    f"{log_prefix} batch {batch_idx}/{len(loader)} | "
                    f"loss={loss.item():.4f} | data_wait={data_wait:.2f}s | "
                    f"step_time={step_time:.2f}s{warn_text}"
                )
            eval_prev_end = time.time()

    avg_loss = running_loss / max(len(loader), 1)
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    return avg_loss, accuracy, all_labels, all_preds


def validate_label_coverage(train_records: List[Dict[str, str]]) -> Tuple[int, int, int]:
    """
    결합 클래스(single-head) 학습을 위한 기본 라벨 커버리지 검증.
    """
    style_count = len({r["style"] for r in train_records})
    gender_count = len({r["gender"] for r in train_records})
    label_count = len({r["label"] for r in train_records})
    if style_count < 2:
        raise ValueError("스타일 클래스가 2개 미만입니다. 학습 데이터 라벨을 확인하세요.")
    if gender_count < 2:
        print("[WARN] train 데이터에 성별이 1개만 있어 결합 클래스 다양성이 제한됩니다.")
    if label_count < 2:
        raise ValueError("결합 클래스(label) 개수가 2개 미만입니다. 파일명 라벨을 확인하세요.")
    return style_count, gender_count, label_count

# def build_model(num_classes: int, pretrained: bool = True):
#     if pretrained:
#         weights = models.ResNet50_Weights.IMAGENET1K_V2
#         model = models.resnet50(weights=weights)
#     else:
#         model = models.resnet50(weights=None)

#     in_features = model.fc.in_features
#     model.fc = nn.Linear(in_features, num_classes)
#     return model

def print_label_mapping(index_to_label: Dict[int, str]) -> None:
    print("\n=== Class Index Mapping ===")
    for idx in range(len(index_to_label)):
        print(f"{idx:2d} -> {index_to_label[idx]}")

def print_split_label_distribution(
    split_name: str,
    records: List[Dict[str, str]],
    label_to_index: Dict[str, int],
) -> None:
    counter = Counter(r["label"] for r in records)

    print(f"\n=== {split_name.upper()} Label Distribution ===")
    print(f"num_samples = {len(records)}")
    print(f"num_classes_present = {len(counter)}")

    for label in sorted(counter.keys(), key=lambda x: label_to_index[x]):
        idx = label_to_index[label]
        count = counter[label]
        print(f"class {idx:2d} ({label}) -> {count}")


def run_single_training(
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, object]:
    """
    단일 모델(single-head) 결합 클래스(style+gender) 학습 파이프라인.
    """
    # STEP 0) 재현성 설정 + 출력 폴더 준비
    seed_everything(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # STEP 1) 파일 스캔 및 라벨 파싱
    train_records, train_invalid = collect_records(args.train_dir)
    if args.val_dir:
        val_records, val_invalid = collect_records(args.val_dir)
    else:
        val_records = []
        val_invalid = []
    if args.test_dir:
        test_records, test_invalid = collect_records(args.test_dir)
    else:
        test_records = []
        test_invalid = []

    if not args.val_dir:
        # val 폴더를 따로 안 주면 train에서 자동 분할
        train_records, val_records = split_train_val_records(
            train_records, args.val_ratio, args.seed
        )

    if len(train_records) == 0 or len(val_records) == 0:
        raise ValueError(
            "Train/validation records are empty. "
            "Check directory and filename format."
        )
    if args.test_dir and len(test_records) == 0:
        raise ValueError("Test records are empty. Check test directory and filename format.")

    # STEP 2) 결합 클래스 라벨 검증
    actual_style_count, actual_gender_count, actual_label_count = validate_label_coverage(train_records)
    print(
        f"[Step 2] Validate labels: style_count={actual_style_count}, "
        f"gender_count={actual_gender_count}, combined_class_count={actual_label_count}"
    )

    # STEP 3) 결합 라벨 문자열 -> 정수 인덱스 사전 생성
    label_list = sorted({r["label"] for r in train_records})
    label_to_index = {label: idx for idx, label in enumerate(label_list)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    print("[Step 3] Build combined label mapping")

    print_label_mapping(index_to_label)

    print_split_label_distribution("train", train_records, label_to_index)
    print_split_label_distribution("val", val_records, label_to_index)
    if len(test_records) > 0:
        print_split_label_distribution("test", test_records, label_to_index)

    # STEP 4) train에 없는 결합 라벨이 val에 있으면 제거
    filtered_val_records = [r for r in val_records if r["label"] in label_to_index]
    dropped_val = len(val_records) - len(filtered_val_records)
    val_records = filtered_val_records
    if len(val_records) == 0:
        raise ValueError("Validation records became empty after label filtering.")
    print("[Step 4] Validate val labels")

    # STEP 4-1) train에 없는 결합 라벨이 test에 있으면 제거
    filtered_test_records = [r for r in test_records if r["label"] in label_to_index]
    dropped_test = len(test_records) - len(filtered_test_records)
    test_records = filtered_test_records
    if args.test_dir and len(test_records) == 0:
        raise ValueError("Test records became empty after label filtering.")

    # STEP 5) Dataset / DataLoader 생성
    train_transform, val_transform = create_transforms(args.image_size)
    train_dataset = FashionStyleDataset(train_records, label_to_index, train_transform)
    val_dataset = FashionStyleDataset(val_records, label_to_index, val_transform)
    test_dataset = (
        FashionStyleDataset(test_records, label_to_index, val_transform)
        if len(test_records) > 0
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if test_dataset is not None
        else None
    )
    print("[Step 5] Create DataSet DataLoader")

    # STEP 6) 모델/손실/옵티마이저 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = resnet50(img_channels=3, num_classes=len(label_list)).to(device)
    model = ResNet50Classifier(
        num_classes=len(label_list),
        dropout=0.2,
        pretrained=True,
    ).to(device)

    feature_dim = model.classifier.in_features # resnet50 -> 2048차원

    class_weights = (
        build_class_weights(train_records, label_to_index, device)
        if args.use_class_weights
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",       # val_loss 기준
        factor=0.5,       # lr *= 0.5
        patience=3,       # 3 epoch 개선 없으면 발동
        min_lr=1e-6,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    print(f"\n[Step 6] Reset ResNet50 Model. Device is {device}")

    print("\n[Step 7] === Start Single Model Training (Combined 31-classes) ===")
    print(f"Num classes: {len(label_list)}, Feature dim: {feature_dim}")
    print(
        f"train_samples={len(train_dataset)}, val_samples={len(val_dataset)}, "
        f"train_batches={len(train_loader)}, val_batches={len(val_loader)}, "
        f"batch_size={args.batch_size}, num_workers={args.num_workers}"
    )
    if test_loader is not None:
        print(
            f"test_samples={len(test_dataset)}, test_batches={len(test_loader)} "
            f"(final evaluation split)"
        )
    else:
        print("[INFO] --test-dir가 없어 final metrics는 validation split 기준으로 계산합니다.")
    if len(train_dataset) > 0:
        print(f"example_train_file={train_dataset.samples[0]['path']}")

    # STEP 7) Epoch 학습 루프 + Early Stopping
    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0
        prev_batch_end = time.time()

        print(f"epoch {epoch + 1}/{args.num_epochs} started (waiting first batch...)")

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            batch_fetch_done = time.time()
            data_wait = batch_fetch_done - prev_batch_end
            step_start = time.time()
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

            if args.log_interval > 0 and (batch_idx % args.log_interval == 0 or batch_idx == len(train_loader)):
                step_time = time.time() - step_start
                current_lr = optimizer.param_groups[0]["lr"]
                samples_per_sec = labels.size(0) / max(step_time, 1e-8)
                warn_text = ""
                if data_wait > args.data_wait_warn_sec:
                    warn_text = f" | [WARN] data_wait>{args.data_wait_warn_sec:.1f}s"
                gpu_mem_text = ""
                if torch.cuda.is_available():
                    gpu_mem_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
                    gpu_mem_text = f" | gpu_mem={gpu_mem_gb:.2f}GB"
                print(
                    f"train epoch {epoch + 1}/{args.num_epochs} "
                    f"batch {batch_idx}/{len(train_loader)} | "
                    f"loss={loss.item():.4f} | lr={current_lr:.6f} | "
                    f"data_wait={data_wait:.2f}s | step_time={step_time:.2f}s | "
                    f"img_per_sec={samples_per_sec:.1f}{gpu_mem_text}{warn_text}"
                )
            prev_batch_end = time.time()

        train_loss = train_running_loss / max(len(train_loader), 1)
        train_acc = (train_correct / train_total) * 100 if train_total > 0 else 0.0
        val_loss, val_acc, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            log_prefix=f"val epoch {epoch + 1}/{args.num_epochs}",
            log_interval=args.log_interval,
            data_wait_warn_sec=args.data_wait_warn_sec,
        )
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch + 1}/{args.num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
            f"epoch_time={epoch_time:.2f}s"
        )
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model_state.pth")
            print(f"best model updated at epoch {epoch + 1} (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(
                f"no improvement ({patience_counter}/{args.patience}) | "
                f"best_val_loss={best_val_loss:.4f}"
            )
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # STEP 8) best 모델 로드 후 final split(val 또는 test) 평가
    best_model_path = output_dir / "best_model_state.pth"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model checkpoint not found: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    final_eval_split = "test" if test_loader is not None else "val"
    final_eval_loader = test_loader if test_loader is not None else val_loader
    final_eval_loss, final_eval_acc_pct, eval_labels, eval_preds = evaluate(
        model=model,
        loader=final_eval_loader,
        criterion=criterion,
        device=device,
        log_prefix=f"[{final_eval_split}] final",
        log_interval=0,
        data_wait_warn_sec=args.data_wait_warn_sec,
    )

    # STEP 9) 최종 평가 지표 계산
    accuracy = accuracy_score(eval_labels, eval_preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        eval_labels,
        eval_preds,
        labels=list(range(len(label_list))),
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        eval_labels,
        eval_preds,
        labels=list(range(len(label_list))),
        average="weighted",
        zero_division=0,
    )

    # STEP 10) confusion matrix / classification report 저장
    cm = confusion_matrix(eval_labels, eval_preds, labels=list(range(len(label_list))))
    target_names = [index_to_label[i] for i in range(len(label_list))]
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in target_names],
        columns=[f"pred_{name}" for name in target_names],
    )
    cm_df.to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    cm_df.to_csv(output_dir / f"{final_eval_split}_confusion_matrix.csv", encoding="utf-8-sig")

    report = classification_report(
        eval_labels,
        eval_preds,
        labels=list(range(len(label_list))),
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    with open(output_dir / f"{final_eval_split}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # STEP 11) 재현/분석용 메타데이터 저장
    save_label_distribution(train_records, output_dir / "train_label_distribution.csv")
    save_label_distribution(val_records, output_dir / "val_label_distribution.csv")
    if len(test_records) > 0:
        save_label_distribution(test_records, output_dir / "test_label_distribution.csv")
    pd.DataFrame(train_records).to_csv(output_dir / "train_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(val_records).to_csv(output_dir / "val_manifest.csv", index=False, encoding="utf-8-sig")
    if len(test_records) > 0:
        pd.DataFrame(test_records).to_csv(output_dir / "test_manifest.csv", index=False, encoding="utf-8-sig")

    with open(output_dir / "label_to_index.json", "w", encoding="utf-8") as f:
        json.dump(label_to_index, f, ensure_ascii=False, indent=2)
    with open(output_dir / "index_to_label.json", "w", encoding="utf-8") as f:
        json.dump(index_to_label, f, ensure_ascii=False, indent=2)

    # STEP 12) 실행 요약 저장
    run_summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_eval_split": final_eval_split,
        "final_eval_loss": final_eval_loss,
        "final_eval_accuracy_percent": final_eval_acc_pct,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "num_train_images": len(train_records),
        "num_val_images": len(val_records),
        "num_test_images": len(test_records),
        "num_classes": len(label_list),
        "feature_dim": feature_dim,
        "actual_style_count": actual_style_count,
        "actual_gender_count": actual_gender_count,
        "actual_label_count": actual_label_count,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "invalid_train_filenames": len(train_invalid),
        "invalid_val_filenames": len(val_invalid),
        "invalid_test_filenames": len(test_invalid),
        "dropped_val_labels_not_in_train": dropped_val,
        "dropped_test_labels_not_in_train": dropped_test,
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    # 참고: 전체 객체 저장은 코드/환경 버전 영향을 받을 수 있습니다.
    # 재로딩 안정성 관점에서는 best_model_state.pth(state_dict)를 우선 권장합니다.
    # torch.save(model, output_dir / "last_model_full.pth")

    print(f"\n=== Final Evaluation ({final_eval_split.upper()}) ===")
    print(f"Best Epoch: {best_epoch}")
    print(f"Final Eval Loss: {final_eval_loss:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(
        f"Macro    - Precision: {precision_macro:.4f}, "
        f"Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}"
    )
    print(
        f"Weighted - Precision: {precision_weighted:.4f}, "
        f"Recall: {recall_weighted:.4f}, F1: {f1_weighted:.4f}"
    )
    print(f"Output directory: {output_dir.resolve()}")

    return run_summary


def run_training(args: argparse.Namespace) -> None:
    """
    단일 모델(single-head) 학습 실행.
    """
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    summary = run_single_training(
        args=args,
        output_dir=base_output_dir,
    )
    with open(base_output_dir / "single_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    """
    명령행 인자 정의.
    팀원이 경로/학습 설정을 코드 수정 없이 바꿀 수 있도록 모든 핵심 옵션을 노출합니다.
    """
    parser = argparse.ArgumentParser(
        description="ResNet50 Fashion Style Training (classification + reusable feature extractor)"
    )
    parser.add_argument("--train-dir", type=str, required=True, help="Train image directory")
    parser.add_argument(
        "--val-dir",
        type=str,
        default="",
        help="Validation image directory. If empty, train-dir에서 자동 분할합니다.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="",
        help="Test image directory. 제공 시 best 모델로 최종 test 평가를 수행합니다.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.3, help="val-dir가 없을 때 자동 분할 비율")
    parser.add_argument("--image-size", type=int, default=224, help="입력 이미지 크기")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size (OOM 시 8/4 권장)")
    parser.add_argument("--num-epochs", type=int, default=30, help="학습 epoch")
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="batch 로그 출력 간격(0이면 batch 로그 비활성화)",
    )
    parser.add_argument(
        "--data-wait-warn-sec",
        type=float,
        default=10.0,
        help="DataLoader 대기시간이 이 값을 넘으면 경고 로그 출력",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--output-dir", type=str, default="outputs_resnet50", help="출력 디렉토리")
    parser.add_argument(
        "--disable-class-weights",
        action="store_true",
        help="클래스 불균형 가중치 비활성화",
    )
    return parser


def main() -> None:
    # 프로그램 시작점: 인자 파싱 후 학습 실행
    parser = build_arg_parser()
    args = parser.parse_args()
    args.use_class_weights = not args.disable_class_weights
    run_training(args)


if __name__ == "__main__":
    main()
