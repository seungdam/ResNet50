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
from torchvision import transforms

# -----------------------------------------------------------------------------
# 프로젝트 개요
# 1) 파일명에서 라벨을 파싱해 학습용 레코드를 구성합니다.
# 2) 결합 라벨(예: casual_male) 기준으로 ResNet50을 학습합니다.
# 3) Accuracy/F1 및 confusion matrix/report를 저장합니다.
# 4) 추천 시스템을 위한 feature extraction 경로를 유지합니다.
# -----------------------------------------------------------------------------


def seed_everything(seed: int = 42) -> None:
    """재현 가능한 실험을 위해 랜덤 시드를 고정합니다."""
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
    파일명에서 메타데이터를 파싱합니다:
    {prefix}_{image_id}_{...}_{style}_{W/M}.jpg

    마지막 토큰 W/M 을 female/male 로 매핑합니다.
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
    image_dir 를 재귀 탐색하여 샘플 레코드를 생성합니다.
    반환값:
    - records: 정상 파싱된 샘플 목록
    - invalid_files: 라벨 파싱 규칙에 맞지 않는 파일 목록
    """
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"이미지 경로를 찾을 수 없습니다: {image_dir}")

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


def print_label_inventory(records: List[Dict[str, str]], title: str) -> None:
    """결합 라벨별 이미지 개수를 출력합니다."""
    label_counter = Counter(rec["label"] for rec in records)
    print(
        f"\n[{title}] total_images={len(records)}, "
        f"unique_combined_labels={len(label_counter)}"
    )
    if not label_counter:
        return
    for idx, label in enumerate(sorted(label_counter.keys()), start=1):
        print(f"  {idx:>2}. {label}: {label_counter[label]}")


def save_label_distribution(records: List[Dict[str, str]], csv_path: Path) -> None:
    """라벨 분포를 CSV로 저장합니다."""
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
    수동 train/validation 분할 함수입니다.
    라벨 그룹별로 독립 분할하여 라벨 비율을 최대한 유지합니다.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if len(records) < 2:
        raise ValueError("Not enough images to split train/val.")

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for rec in records:
        grouped.setdefault(rec["label"], []).append(rec)

    rng = random.Random(seed)
    train_records: List[Dict[str, str]] = []
    val_records: List[Dict[str, str]] = []
    low_count_labels: List[str] = []

    for label, group in grouped.items():
        group = group.copy()
        rng.shuffle(group)
        n_total = len(group)
        if n_total < 2:
            # 샘플이 1개뿐인 라벨은 train/validation 분할이 불가능합니다.
            train_records.extend(group)
            low_count_labels.append(label)
            continue

        n_val = int(round(n_total * val_ratio))
        n_val = min(max(n_val, 1), n_total - 1)
        n_train = n_total - n_val
        train_records.extend(group[:n_train])
        val_records.extend(group[n_train:])

    if len(val_records) == 0:
        raise ValueError("분할 후 validation 레코드가 비어 있습니다.")
    if low_count_labels:
        print(
            f"[WARN] 샘플이 2개 미만이라 분할 불가한 라벨: {len(low_count_labels)}개. "
            "해당 라벨은 train에만 포함됩니다."
        )

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def split_train_val_records_stratified(
    records: List[Dict[str, str]], val_ratio: float, seed: int
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """결합 라벨(style_gender) 기준 sklearn stratified split."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if len(records) < 2:
        raise ValueError("Not enough images to split train/val.")

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
    """이미지 텐서와 결합 라벨 인덱스를 반환하는 Dataset."""

    def __init__(
        self,
        records: List[Dict[str, str]],
        label_to_index: Dict[str, int],
        transform: Optional[transforms.Compose] = None,
        image_size: int = 224,
    ):
        self.transform = transform
        # 전처리 단계에서 이미 필터링된 records를 사용합니다.
        self.samples = list(records)
        self.label_to_index = label_to_index
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        item = self.samples[idx]
        try:
            image = Image.open(item["path"]).convert("RGB")
        except Exception as e:
            print(f"[WARN] 이미지 로드 실패, fallback 이미지를 사용합니다: {item['path']} | {e}")
            # fallback 이미지는 3:4 세로 비율을 유지합니다.
            target_h = int(round(self.image_size * (4 / 3)))
            image = Image.new("RGB", (self.image_size, target_h), color=0)
        if self.transform is not None:
            image = self.transform(image)
        label_idx = self.label_to_index[item["label"]]
        return image, label_idx


class Bottleneck(nn.Module):
    # ResNet50 bottleneck block: 1x1 -> 3x3 -> 1x1, expansion=4
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 4,
        downsample: nn.Module = None,
    ) -> None:
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet 본체 모델입니다.
    - forward(): 분류 logits를 반환
    - extract_feature_vector(): 검색/추천용 임베딩 벡터를 반환
    """

    def __init__(
        self,
        img_channels: int,
        num_layers: int,
        block: Type[nn.Module],
        num_classes: int = 1000,
        dropout_p: float = 0.0,
    ) -> None:
        super(ResNet, self).__init__()
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError("dropout_p must be in [0.0, 1.0).")
        if num_layers == 50:
            layers = [3, 4, 6, 3]
            self.expansion = 4
        else:
            raise ValueError("num_layers must be 50.")

        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7, stride=2, padding=3, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # dropout은 FC head 직전에만 적용합니다.
        # FAISS 인덱싱 안정성을 위해 feature는 dropout 이전 벡터를 사용합니다.
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()
        self.fc = nn.Linear(512 * self.expansion, num_classes)

        # He 초기화는 scratch 학습 안정성에 도움이 됩니다.
        self._initialize_weights()

    def _make_layer(
        self,
        block: Type[nn.Module],
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * self.expansion,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

        layers = [
            block(self.in_channels, out_channels, stride, self.expansion, downsample)
        ]
        self.in_channels = out_channels * self.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, expansion=self.expansion))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """ReLU 기반 Conv layer에 He 초기화를 적용합니다."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_backbone(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_head(self, feature_map: Tensor) -> Tuple[Tensor, Tensor]:
        pooled = self.avgpool(feature_map)
        feature_vector = torch.flatten(pooled, 1)
        # FC 직전에만 dropout을 적용하고, 원본 feature vector는 분리 유지합니다.
        logits = self.fc(self.dropout(feature_vector))
        return logits, feature_vector

    def extract_feature_vector(self, x: Tensor, normalize: bool = True) -> Tensor:
        """추천/검색 추론에 사용할 feature vector를 추출합니다."""
        was_training = self.training
        self.eval()
        with torch.inference_mode():
            feature_map = self.forward_backbone(x)
            _, feature_vector = self.forward_head(feature_map)
        if was_training:
            self.train()
        if normalize:
            feature_vector = F.normalize(feature_vector, p=2, dim=1)
        return feature_vector

    def forward(
        self,
        x: Tensor,
        return_features: bool = False,
        return_feature_map: bool = False,
    ):
        feature_map = self.forward_backbone(x)
        logits, feature_vector = self.forward_head(feature_map)
        if return_features and return_feature_map:
            return logits, feature_vector, feature_map
        if return_features:
            return logits, feature_vector
        return logits


def resnet50(img_channels: int, num_classes: int, dropout_p: float = 0.0) -> ResNet:
    """ResNet50 생성 헬퍼 함수."""
    return ResNet(img_channels, 50, Bottleneck, num_classes, dropout_p=dropout_p)


def create_transforms(
    image_size: int,
    profile: str = "legacy",
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    train/validation 변환을 생성합니다.
    - train: augmentation + normalize
    - val: resize + normalize

    legacy 프로필은 3000x4000 세로형 이미지 비율 특성을 반영합니다.
    team 프로필은 ImageNet 호환 정규화를 사용합니다.
    """
    # legacy 모드는 세로(H)가 더 긴 형태를 유지합니다.
    if profile == "team":
        train_transform = transforms.Compose([
            transforms.Resize((2 * image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
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

    if profile != "legacy":
        raise ValueError(f"알 수 없는 transform profile: {profile}. ['legacy', 'team'] 중 하나를 사용하세요.")

    target_h = int(round(image_size * (4 / 3)))  # 299
    target_w = image_size                          # 224

    train_transform = transforms.Compose([
        transforms.Resize((target_h, target_w)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5498, 0.5226, 0.5052],
            std=[0.2600, 0.2582, 0.2620],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((target_h, target_w)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5498, 0.5226, 0.5052],
            std=[0.2600, 0.2582, 0.2620],
        ),
    ])

    return train_transform, val_transform


def build_class_weights(
    train_records: List[Dict[str, str]],
    label_to_index: Dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    """
    불균형 라벨 보정을 위한 역빈도 class weight를 계산합니다.
    샘플 수가 적은 클래스일수록 더 큰 가중치를 받습니다.
    """
    class_counts = np.zeros(len(label_to_index), dtype=np.float32)
    for rec in train_records:
        class_counts[label_to_index[rec["label"]]] += 1.0

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
    주어진 loader에서 모델을 평가하고 예측 결과를 수집합니다.
    반환된 labels/preds는 confusion matrix와 F1 계산에 사용됩니다.
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
                    f"{log_prefix} 배치 {batch_idx}/{len(loader)} | "
                    f"손실={loss.item():.4f} | data_wait={data_wait:.2f}s | "
                    f"step_time={step_time:.2f}s{warn_text}"
                )
            eval_prev_end = time.time()

    avg_loss = running_loss / max(len(loader), 1)
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    return avg_loss, accuracy, all_labels, all_preds


def validate_label_coverage(train_records: List[Dict[str, str]]) -> Tuple[int, int, int]:
    """결합 라벨 single-head 학습을 위한 최소 라벨 커버리지를 검증합니다."""
    style_count = len({r["style"] for r in train_records})
    gender_count = len({r["gender"] for r in train_records})
    label_count = len({r["label"] for r in train_records})
    if style_count < 2:
        raise ValueError("학습 데이터에 style 클래스가 최소 2개 이상 필요합니다.")
    if gender_count < 2:
        print("[WARN] train 데이터에 성별이 1개만 존재합니다. 결합 라벨 다양성이 제한됩니다.")
    if label_count < 2:
        raise ValueError("학습 데이터에 결합 라벨이 최소 2개 이상 필요합니다.")
    return style_count, gender_count, label_count


def run_single_training(
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, object]:
    """결합 라벨(style_gender) 기준 single-head 학습을 수행합니다."""
    if not 0.0 <= args.dropout_p < 1.0:
        raise ValueError("--dropout-p must be in [0.0, 1.0).")

    if args.compare_with_team:
        # 팀 설정과 공정 비교를 위한 원스위치 프리셋
        args.split_mode = "stratified"
        args.transform_profile = "team"
        print(
            "[Preset] compare-with-team 활성화: split=stratified, transform=team "
            "(필요하면 batch/lr/dropout을 팀 설정으로 맞추세요)"
        )

    # STEP 0) 재현성 및 출력 경로 준비
    seed_everything(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # STEP 1) 파일/라벨 파싱
    all_records, train_invalid = collect_records(args.train_dir)
    print_label_inventory(all_records, "원본 train_dir 라벨 분포 (분할 전)")

    # 선택한 split 모드와 val_ratio로 분할
    if args.split_mode == "manual":
        train_records, val_records = split_train_val_records(
            all_records, val_ratio=args.val_ratio, seed=args.seed
        )
    elif args.split_mode == "stratified":
        train_records, val_records = split_train_val_records_stratified(
            all_records, val_ratio=args.val_ratio, seed=args.seed
        )
    else:
        raise ValueError(f"알 수 없는 split mode: {args.split_mode}")
    val_invalid: List[str] = []  # val은 train 분할로 생성되므로 별도 invalid 스캔 없음
    print_label_inventory(train_records, f"Train 분할 ({1 - args.val_ratio:.0%}/{args.val_ratio:.0%})")
    print_label_inventory(val_records,   f"Val 분할   ({1 - args.val_ratio:.0%}/{args.val_ratio:.0%})")

    # 선택적 test_dir 처리 (--test-dir)
    if args.test_dir:
        test_records, test_invalid = collect_records(args.test_dir)
        print_label_inventory(test_records, "Test 라벨 분포")
    else:
        test_records = []
        test_invalid = []

    if len(train_records) == 0 or len(val_records) == 0:
        raise ValueError("Train/validation 레코드가 비어 있습니다. 경로와 파일명 형식을 확인하세요.")
    if args.test_dir and len(test_records) == 0:
        raise ValueError("Test 레코드가 비어 있습니다. test 경로와 파일명 형식을 확인하세요.")

    # STEP 2) 라벨 커버리지 검증
    actual_style_count, actual_gender_count, actual_label_count = validate_label_coverage(train_records)
    print(
        f"[Step 2] style_count={actual_style_count}, "
        f"gender_count={actual_gender_count}, combined_class_count={actual_label_count}"
    )

    # STEP 3) label -> index 매핑 생성
    label_list = sorted({r["label"] for r in train_records})
    label_to_index = {label: idx for idx, label in enumerate(label_list)}
    # JSON 저장/로드 후 key가 문자열로 바뀔 수 있어 메모리에서는 int key를 유지합니다.
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    print("[Step 3] 결합 라벨 매핑 생성")
    for idx in range(len(label_list)):
        print(f"  class[{idx}] = {index_to_label[idx]}")

    # STEP 4) train에 없는 라벨을 val/test에서 제거
    filtered_val_records = [r for r in val_records if r["label"] in label_to_index]
    dropped_val = len(val_records) - len(filtered_val_records)
    val_records = filtered_val_records
    if len(val_records) == 0:
        raise ValueError("라벨 필터링 후 Validation 레코드가 비었습니다.")
    if dropped_val > 0:
        print(f"[WARN] train에 없는 라벨의 validation 샘플 {dropped_val}개를 제거했습니다.")

    filtered_test_records = [r for r in test_records if r["label"] in label_to_index]
    dropped_test = len(test_records) - len(filtered_test_records)
    test_records = filtered_test_records
    if args.test_dir and len(test_records) == 0:
        raise ValueError("라벨 필터링 후 Test 레코드가 비었습니다.")

    # STEP 5) Dataset / DataLoader 생성
    train_transform, val_transform = create_transforms(
        args.image_size,
        profile=args.transform_profile,
    )
    train_dataset = FashionStyleDataset(train_records, label_to_index, train_transform, args.image_size)
    val_dataset   = FashionStyleDataset(val_records,   label_to_index, val_transform,   args.image_size)
    test_dataset  = (
        FashionStyleDataset(test_records, label_to_index, val_transform, args.image_size)
        if len(test_records) > 0 else None
    )

    # 불균형 보정은 class_weights만 사용합니다.

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
        if test_dataset is not None else None
    )
    print("[Step 5] DataSet/DataLoader 생성 (shuffle=True)")

    # STEP 6) 모델 / 손실함수 / 옵티마이저 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 내부 custom ResNet50만 사용합니다 (scratch 학습 고정).
    model = resnet50(
        img_channels=3,
        num_classes=len(label_list),
        dropout_p=args.dropout_p,
    ).to(device)

    feature_dim = int(model.fc.in_features)  # custom ResNet50 = 2048

    apply_class_weights = args.use_class_weights
    class_weights = (
        build_class_weights(train_records, label_to_index, device)
        if apply_class_weights else None
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # AdamW는 현재 설정에서 안정적인 기본 optimizer입니다.
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    print(
        f"\n[Step 6] ResNet50 초기화 완료. Device={device}, "
        f"init_mode=scratch, dropout_p={args.dropout_p}"
    )

    print("\n[Step 7] === Single-Head 학습 시작 (결합 31클래스) ===")
    print(f"클래스 수: {len(label_list)}, Feature 차원: {feature_dim}")
    print(
        f"train 샘플={len(train_dataset)}, val 샘플={len(val_dataset)}, "
        f"train 배치={len(train_loader)}, val 배치={len(val_loader)}, "
        f"batch_size={args.batch_size}, num_workers={args.num_workers}, "
        f"val_ratio={args.val_ratio}, split_mode={args.split_mode}, "
        f"transform_profile={args.transform_profile}, init_mode=scratch, "
        f"dropout_p={args.dropout_p}, lr={args.lr}, "
        f"use_class_weights={apply_class_weights}"
    )
    if test_loader is not None:
        print(f"test 샘플={len(test_dataset)}, test 배치={len(test_loader)} (최종 평가)")
    else:
        print("[INFO] --test-dir 미입력: 최종 지표는 validation split 기준으로 계산합니다.")
    if len(train_dataset) > 0:
        print(f"예시 train 파일={train_dataset.samples[0]['path']}")

    # STEP 7) Epoch 학습 루프 + Early Stopping
    for epoch in range(args.num_epochs):
        model.train()
        epoch_start = time.time()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0
        prev_batch_end = time.time()

        print(f"epoch {epoch + 1}/{args.num_epochs} 시작 (첫 배치 대기 중...)")

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
            # Gradient clipping은 학습 안정화에 도움이 됩니다.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

            if args.log_interval > 0 and (
                batch_idx % args.log_interval == 0 or batch_idx == len(train_loader)
            ):
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
                    f"배치 {batch_idx}/{len(train_loader)} | "
                    f"손실={loss.item():.4f} | lr={current_lr:.6f} | "
                    f"data_wait={data_wait:.2f}s | step_time={step_time:.2f}s | "
                    f"img_per_sec={samples_per_sec:.1f}{gpu_mem_text}{warn_text}"
                )
            prev_batch_end = time.time()

        train_loss = train_running_loss / max(len(train_loader), 1)
        train_acc  = (train_correct / train_total) * 100 if train_total > 0 else 0.0
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
            f"Train 손실: {train_loss:.4f}, Train 정확도: {train_acc:.2f}% | "
            f"Val 손실: {val_loss:.4f}, Val 정확도: {val_acc:.2f}% | "
            f"epoch_time={epoch_time:.2f}s"
        )

        # Scheduler는 val_loss 기준으로 step을 진행합니다.
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model_state.pth")
            print(f"epoch {epoch + 1}에서 best model 갱신 (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(
                f"성능 개선 없음 ({patience_counter}/{args.patience}) | "
                f"best_val_loss={best_val_loss:.4f}"
            )
            if patience_counter >= args.patience:
                print(f"epoch {epoch + 1}에서 Early stopping")
                break

    # STEP 8) best 체크포인트 로드 후 최종 split(val 또는 test) 평가
    best_model_path = output_dir / "best_model_state.pth"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model checkpoint not found: {best_model_path}")
    # weights_only=True로 전체 객체 로드 이슈를 줄입니다.
    model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=True)
    )

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

    # STEP 9) 최종 지표 계산
    accuracy = accuracy_score(eval_labels, eval_preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        eval_labels, eval_preds,
        labels=list(range(len(label_list))), average="macro", zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        eval_labels, eval_preds,
        labels=list(range(len(label_list))), average="weighted", zero_division=0,
    )

    # STEP 10) confusion matrix / classification report 저장
    # 선택된 최종 split에 대해 단일 결과를 저장합니다.
    cm = confusion_matrix(eval_labels, eval_preds, labels=list(range(len(label_list))))
    target_names = [index_to_label[i] for i in range(len(label_list))]
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{n}" for n in target_names],
        columns=[f"pred_{n}" for n in target_names],
    )
    cm_df.to_csv(output_dir / f"{final_eval_split}_confusion_matrix.csv", encoding="utf-8-sig")

    report = classification_report(
        eval_labels, eval_preds,
        labels=list(range(len(label_list))),
        target_names=target_names,
        digits=4, zero_division=0,
    )
    with open(output_dir / f"{final_eval_split}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # STEP 11) 메타데이터/manifest 저장
    save_label_distribution(train_records, output_dir / "train_label_distribution.csv")
    save_label_distribution(val_records,   output_dir / "val_label_distribution.csv")
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

    # STEP 12) 실행 요약(run summary) 구성
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
        "model_family": "resnet50_custom",
        "init_mode": "scratch",
        "split_mode": args.split_mode,
        "transform_profile": args.transform_profile,
        "compare_with_team": args.compare_with_team,
        "val_ratio": args.val_ratio,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "dropout_p": args.dropout_p,
        "lr": args.lr,
        "use_class_weights": apply_class_weights,
        "invalid_train_filenames": len(train_invalid),
        "invalid_val_filenames": 0,  # val은 train_dir 분할로 생성
        "invalid_test_filenames": len(test_invalid) if args.test_dir else None,
        "dropped_val_labels_not_in_train": dropped_val,
        "dropped_test_labels_not_in_train": dropped_test,
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    # 편의를 위해 full model도 저장합니다. 공유/이식에는 state_dict 권장을 유지합니다.
    torch.save(model, output_dir / "last_model_full.pth")

    print(f"\n=== 최종 평가 ({final_eval_split.upper()}) ===")
    print(f"최적 Epoch: {best_epoch}")
    print(f"최종 Eval Loss: {final_eval_loss:.4f}")
    print(f"정확도(Accuracy): {accuracy * 100:.2f}%")
    print(f"Macro    - Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")
    print(f"Weighted - Precision: {precision_weighted:.4f}, Recall: {recall_weighted:.4f}, F1: {f1_weighted:.4f}")
    print(f"출력 경로: {output_dir.resolve()}")

    return run_summary


def run_training(args: argparse.Namespace) -> None:
    """single-head 학습 진입점 함수."""
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    summary = run_single_training(args=args, output_dir=base_output_dir)
    with open(base_output_dir / "single_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    """명령행 인자를 정의합니다."""
    parser = argparse.ArgumentParser(
        description="ResNet50 패션 스타일 학습 (single-head, scratch 고정)"
    )
    parser.add_argument("--train-dir", type=str, required=True, help="학습 이미지 경로")
    parser.add_argument("--test-dir", type=str, default="", help="테스트 이미지 경로 (선택)")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.3,
        help="Validation 분할 비율 (기본 0.3 -> 70/30 분할)",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["manual", "stratified"],
        default="manual",
        help="Train/Val 분할 방식: manual(라벨별) 또는 sklearn stratified",
    )
    parser.add_argument(
        "--transform-profile",
        type=str,
        choices=["legacy", "team"],
        default="legacy",
        help="Transform 프로필: legacy 또는 team",
    )
    parser.add_argument(
        "--compare-with-team",
        action="store_true",
        help="공정 비교 프리셋: stratified split + team transform",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="입력 이미지 가로 크기 (legacy는 세로 비율을 유지)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="배치 크기 (OOM 발생 시 16/8로 축소)",
    )
    parser.add_argument("--num-epochs", type=int, default=50, help="최대 학습 epoch 수")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW 학습률")
    parser.add_argument(
        "--dropout-p",
        type=float,
        default=0.5,
        help="분류기 head 직전 Dropout 비율",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker 수 (Windows는 0부터 시작 권장)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="배치 로그 출력 간격 (0이면 비활성화)",
    )
    parser.add_argument(
        "--data-wait-warn-sec",
        type=float,
        default=10.0,
        help="data loading 대기시간 경고 임계값(초)",
    )
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--output-dir", type=str, default="outputs_resnet50", help="출력 디렉터리")
    parser.add_argument(
        "--disable-class-weights",
        action="store_true",
        help="CrossEntropyLoss의 class weights 비활성화",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.use_class_weights = not args.disable_class_weights
    run_training(args)


if __name__ == "__main__":
    main()
