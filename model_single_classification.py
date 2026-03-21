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
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

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


def print_label_inventory(records: List[Dict[str, str]], title: str) -> None:
    """
    결합 라벨(label=style_gender) 목록과 각 라벨별 이미지 개수를 출력합니다.
    라벨 파싱/분할이 의도대로 되었는지 초반 점검용으로 사용합니다.
    """
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
    """
    라벨 분포를 CSV로 저장합니다.
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
    라벨별(label 기준)로 val_ratio 비율에 맞춰 train/val을 직접 분할합니다.
    sklearn stratify와 달리 각 클래스에서 정확한 비율이 보장됩니다.
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
            # 샘플 1개 클래스는 validation으로 분할할 수 없어 train에만 배치
            train_records.extend(group)
            low_count_labels.append(label)
            continue

        n_val = int(round(n_total * val_ratio))
        n_val = min(max(n_val, 1), n_total - 1)
        n_train = n_total - n_val
        train_records.extend(group[:n_train])
        val_records.extend(group[n_train:])

    if len(val_records) == 0:
        raise ValueError("Validation records are empty after split.")
    if low_count_labels:
        print(
            f"[WARN] validation 분할 불가(샘플<2) 라벨 수: {len(low_count_labels)} | "
            "해당 라벨은 train에만 배치되었습니다."
        )

    rng.shuffle(train_records)
    rng.shuffle(val_records)
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
        image_size: int = 224,
    ):
        self.transform = transform
        # STEP 4에서 이미 필터링 완료된 records를 그대로 사용
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
            print(f"[WARN] 이미지 로드 실패, 검은 이미지 대체: {item['path']} | {e}")
            # 원본 비율(3:4 portrait) 유지한 fallback 이미지
            target_h = int(round(self.image_size * (4 / 3)))
            image = Image.new("RGB", (self.image_size, target_h), color=0)
        if self.transform is not None:
            image = self.transform(image)
        label_idx = self.label_to_index[item["label"]]
        return image, label_idx


class BasicBlock(nn.Module):
    """
    ResNet18 계열에서 주로 사용하는 기본 residual block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels * self.expansion,
            kernel_size=3, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    # ResNet50의 핵심 블록: 1x1 -> 3x3 -> 1x1, 마지막에 채널 4배 확장
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
    ResNet 본체.
    - forward(): 분류 logits 반환
    - extract_feature_vector(): 추천 시스템에서 쓸 임베딩 벡터 반환
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
        if num_layers == 18:
            layers = [2, 2, 2, 2]
            self.expansion = 1
        elif num_layers == 50:
            layers = [3, 4, 6, 3]
            self.expansion = 4
        else:
            raise ValueError("num_layers must be one of [18, 50].")

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
        # dropout은 fc 직전에만 적용.
        # feature_vector(추천용 임베딩)는 dropout 전 원본을 반환해 FAISS 일관성 보장.
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()
        self.fc = nn.Linear(512 * self.expansion, num_classes)

        # He 초기화: scratch 학습 수렴 안정화
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
        """He 초기화: ReLU 계열 scratch 학습에 최적화된 표준 초기화."""
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
        # dropout은 fc 직전, feature_vector는 dropout 전 원본 반환
        logits = self.fc(self.dropout(feature_vector))
        return logits, feature_vector

    def extract_feature_vector(self, x: Tensor, normalize: bool = True) -> Tensor:
        """추천 시스템에서 사용할 임베딩 추출 함수."""
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
    """편의 함수: ResNet50 인스턴스를 생성."""
    return ResNet(img_channels, 50, Bottleneck, num_classes, dropout_p=dropout_p)


def resnet18(img_channels: int, num_classes: int, dropout_p: float = 0.0) -> ResNet:
    return ResNet(img_channels, 18, BasicBlock, num_classes, dropout_p=dropout_p)


def create_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    학습/검증 전처리 파이프라인 구성.
    - train: augmentation 포함
    - val: resize + normalize 중심

    원본 이미지 비율: 3000(W) x 4000(H) = 3:4 세로형(portrait).
    → target_h > target_w 로 세로를 길게 유지해야 비율 왜곡 없음.
    예: image_size=224 → (target_h=299, target_w=224)
    """
    # [수정] 가로/세로 비율 올바르게 적용: 세로(H)가 더 길어야 portrait 유지
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
    클래스 불균형 완화를 위한 손실 가중치 계산.
    샘플이 적은 클래스일수록 더 큰 가중치가 부여됩니다.
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
    """결합 클래스(single-head) 학습을 위한 기본 라벨 커버리지 검증."""
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


def run_single_training(
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, object]:
    """
    단일 모델(single-head) 결합 클래스(style+gender) 학습 파이프라인.
    """
    if not 0.0 <= args.dropout_p < 1.0:
        raise ValueError("--dropout-p must be in [0.0, 1.0).")

    # STEP 0) 재현성 설정 + 출력 폴더 준비
    seed_everything(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # STEP 1) 파일 스캔 및 라벨 파싱
    all_records, train_invalid = collect_records(args.train_dir)
    print_label_inventory(all_records, "Raw train_dir label inventory (before split)")

    # 라벨별 정확한 val_ratio 비율로 분할
    train_records, val_records = split_train_val_records(
        all_records, val_ratio=args.val_ratio, seed=args.seed
    )
    val_invalid: List[str] = []  # val은 train_dir에서 분할, 별도 invalid 없음
    print_label_inventory(train_records, f"Train split (after {1 - args.val_ratio:.0%}/{args.val_ratio:.0%} split)")
    print_label_inventory(val_records,   f"Val split   (after {1 - args.val_ratio:.0%}/{args.val_ratio:.0%} split)")

    # test_dir 처리 (--test-dir 인자)
    if args.test_dir:
        test_records, test_invalid = collect_records(args.test_dir)
        print_label_inventory(test_records, "Test label inventory")
    else:
        test_records = []
        test_invalid = []

    if len(train_records) == 0 or len(val_records) == 0:
        raise ValueError("Train/validation records are empty. Check directory and filename format.")
    if args.test_dir and len(test_records) == 0:
        raise ValueError("Test records are empty. Check test directory and filename format.")

    # STEP 2) 결합 클래스 라벨 검증
    actual_style_count, actual_gender_count, actual_label_count = validate_label_coverage(train_records)
    print(
        f"[Step 2] style_count={actual_style_count}, "
        f"gender_count={actual_gender_count}, combined_class_count={actual_label_count}"
    )

    # STEP 3) 결합 라벨 문자열 -> 정수 인덱스 사전 생성
    label_list = sorted({r["label"] for r in train_records})
    label_to_index = {label: idx for idx, label in enumerate(label_list)}
    # JSON 저장 시 key가 str로 변환되므로 처음부터 str로 통일
    index_to_label = {str(idx): label for label, idx in label_to_index.items()}
    print("[Step 3] Build combined label mapping")
    for idx in range(len(label_list)):
        print(f"  class[{idx}] = {index_to_label[str(idx)]}")

    # STEP 4) train에 없는 결합 라벨이 val/test에 있으면 제거
    filtered_val_records = [r for r in val_records if r["label"] in label_to_index]
    dropped_val = len(val_records) - len(filtered_val_records)
    val_records = filtered_val_records
    if len(val_records) == 0:
        raise ValueError("Validation records became empty after label filtering.")
    if dropped_val > 0:
        print(f"[WARN] val에서 train에 없는 라벨 {dropped_val}개 제거됨")

    filtered_test_records = [r for r in test_records if r["label"] in label_to_index]
    dropped_test = len(test_records) - len(filtered_test_records)
    test_records = filtered_test_records
    if args.test_dir and len(test_records) == 0:
        raise ValueError("Test records became empty after label filtering.")

    # STEP 5) Dataset / DataLoader 생성
    train_transform, val_transform = create_transforms(args.image_size)
    train_dataset = FashionStyleDataset(train_records, label_to_index, train_transform, args.image_size)
    val_dataset   = FashionStyleDataset(val_records,   label_to_index, val_transform,   args.image_size)
    test_dataset  = (
        FashionStyleDataset(test_records, label_to_index, val_transform, args.image_size)
        if len(test_records) > 0 else None
    )

    # 불균형 보정은 "sampler" 또는 "class_weights" 중 하나만 적용(이중 보정 방지)
    use_sampler = not args.disable_sampler
    sampler = None
    train_shuffle = True
    if use_sampler:
        label_counter = Counter(r["label"] for r in train_dataset.samples)
        sample_weights = [
            1.0 / label_counter[r["label"]] for r in train_dataset.samples
        ]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=train_shuffle,
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
    if use_sampler:
        print("[Step 5] Create DataSet / DataLoader (WeightedRandomSampler 적용)")
    else:
        print("[Step 5] Create DataSet / DataLoader (shuffle=True, sampler 미사용)")

    # STEP 6) 모델/손실/옵티마이저 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(
        img_channels=3,
        num_classes=len(label_list),
        dropout_p=args.dropout_p,
    ).to(device)
    feature_dim = int(model.fc.in_features)  # ResNet50 = 2048

    apply_class_weights = args.use_class_weights and (not use_sampler)
    if args.use_class_weights and use_sampler:
        print(
            "[INFO] sampler와 class_weights 동시 적용은 과보정 위험이 있어 "
            "class_weights를 자동 비활성화합니다."
        )
    class_weights = (
        build_class_weights(train_records, label_to_index, device)
        if apply_class_weights else None
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # AdamW: scratch 학습에서 weight_decay가 실질적으로 동작하는 optimizer
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
    print(f"\n[Step 6] ResNet50 initialized. Device={device}, dropout_p={args.dropout_p}")

    print("\n[Step 7] === Start Single-Head Training (Combined 31-classes) ===")
    print(f"Num classes: {len(label_list)}, Feature dim: {feature_dim}")
    print(
        f"train_samples={len(train_dataset)}, val_samples={len(val_dataset)}, "
        f"train_batches={len(train_loader)}, val_batches={len(val_loader)}, "
        f"batch_size={args.batch_size}, num_workers={args.num_workers}, "
        f"val_ratio={args.val_ratio}, dropout_p={args.dropout_p}, lr={args.lr}, "
        f"use_sampler={use_sampler}, use_class_weights={apply_class_weights}"
    )
    if test_loader is not None:
        print(f"test_samples={len(test_dataset)}, test_batches={len(test_loader)} (final eval)")
    else:
        print("[INFO] --test-dir 없음. final metrics는 val split 기준으로 계산합니다.")
    if len(train_dataset) > 0:
        print(f"example_train_file={train_dataset.samples[0]['path']}")

    # STEP 7) Epoch 학습 루프 + Early Stopping
    for epoch in range(args.num_epochs):
        model.train()
        epoch_start = time.time()
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
            # gradient clipping: scratch 초기 학습의 gradient 폭발 방지
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
                    f"batch {batch_idx}/{len(train_loader)} | "
                    f"loss={loss.item():.4f} | lr={current_lr:.6f} | "
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
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
            f"epoch_time={epoch_time:.2f}s"
        )

        # scheduler는 val_loss 기준, early stopping 판단 전에 호출
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
    # weights_only=True: PyTorch 2.0+ 경고 방지 및 보안 강화
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

    # STEP 9) 최종 평가 지표 계산
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
    # split 이름이 붙은 파일 하나만 저장 (중복 제거)
    cm = confusion_matrix(eval_labels, eval_preds, labels=list(range(len(label_list))))
    target_names = [index_to_label[str(i)] for i in range(len(label_list))]
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

    # STEP 11) 재현/분석용 메타데이터 저장
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
        "val_ratio": args.val_ratio,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "dropout_p": args.dropout_p,
        "lr": args.lr,
        "use_sampler": use_sampler,
        "use_class_weights": apply_class_weights,
        "invalid_train_filenames": len(train_invalid),
        "invalid_val_filenames": 0,  # val은 train_dir에서 분할, 별도 스캔 없음
        "invalid_test_filenames": len(test_invalid) if args.test_dir else None,
        "dropped_val_labels_not_in_train": dropped_val,
        "dropped_test_labels_not_in_train": dropped_test,
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    # state_dict 우선 권장. full 객체 저장은 환경 의존성 있음.
    torch.save(model, output_dir / "last_model_full.pth")

    print(f"\n=== Final Evaluation ({final_eval_split.upper()}) ===")
    print(f"Best Epoch: {best_epoch}")
    print(f"Final Eval Loss: {final_eval_loss:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro    - Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")
    print(f"Weighted - Precision: {precision_weighted:.4f}, Recall: {recall_weighted:.4f}, F1: {f1_weighted:.4f}")
    print(f"Output directory: {output_dir.resolve()}")

    return run_summary


def run_training(args: argparse.Namespace) -> None:
    """단일 모델(single-head) 학습 실행."""
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    summary = run_single_training(args=args, output_dir=base_output_dir)
    with open(base_output_dir / "single_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    """
    명령행 인자 정의.
    팀원이 경로/학습 설정을 코드 수정 없이 바꿀 수 있도록 모든 핵심 옵션을 노출합니다.
    """
    parser = argparse.ArgumentParser(
        description="ResNet50 Fashion Style Training (single-head, scratch)"
    )
    parser.add_argument("--train-dir",  type=str, required=True, help="Train image directory")
    parser.add_argument("--test-dir",   type=str, default="",    help="Test image directory (optional)")
    parser.add_argument(
        "--val-ratio", type=float, default=0.3,
        help="라벨별 validation 분할 비율 (기본 0.3 → 각 라벨 70/30)",
    )
    parser.add_argument("--image-size",  type=int,   default=224,  help="입력 이미지 단변(width) 크기")
    parser.add_argument("--batch-size",  type=int,   default=32,   help="batch size (OOM 시 16/8 권장)")
    parser.add_argument("--num-epochs",  type=int,   default=50,   help="최대 학습 epoch")
    parser.add_argument("--patience",    type=int,   default=7,    help="early stopping patience")
    parser.add_argument("--lr",          type=float, default=3e-4, help="learning rate (AdamW)")
    parser.add_argument(
        "--dropout-p", type=float, default=0.5,
        help="fc 직전 dropout 비율 (기본 0.5, 소규모 데이터 0.3~0.4 권장)",
    )
    parser.add_argument("--num-workers", type=int, default=0,  help="DataLoader workers (Windows=0 권장)")
    parser.add_argument("--log-interval", type=int, default=20, help="batch 로그 출력 간격 (0=비활성화)")
    parser.add_argument("--data-wait-warn-sec", type=float, default=10.0, help="DataLoader 대기 경고 임계값(초)")
    parser.add_argument("--seed",        type=int,   default=42,   help="random seed")
    parser.add_argument("--output-dir",  type=str,   default="outputs_resnet50", help="출력 디렉토리")
    parser.add_argument(
        "--disable-class-weights", action="store_true",
        help="CrossEntropyLoss class_weights 비활성화",
    )
    parser.add_argument(
        "--disable-sampler", action="store_true",
        help="WeightedRandomSampler 비활성화 (이 경우 shuffle=True로 학습)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.use_class_weights = not args.disable_class_weights
    run_training(args)


if __name__ == "__main__":
    main()
