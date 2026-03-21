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
from torch.utils.data import DataLoader, Dataset
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
    라벨별(label 기준)로 7:3 비율에 맞춰 train/val을 직접 분할합니다.
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
    한 샘플을 꺼낼 때
    (이미지 텐서, style 인덱스, gender 인덱스, combined 인덱스)를 반환합니다.
    """

    def __init__(
        self,
        records: List[Dict[str, str]],
        style_to_index: Dict[str, int],
        gender_to_index: Dict[str, int],
        label_to_index: Dict[str, int],
        transform: Optional[transforms.Compose] = None,
        image_size: int = 224,
    ):
        self.transform = transform
        self.samples = [
            r
            for r in records
            if r["style"] in style_to_index
            and r["gender"] in gender_to_index
            and r["label"] in label_to_index
        ]
        self.style_to_index = style_to_index
        self.gender_to_index = gender_to_index
        self.label_to_index = label_to_index
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int, int, int]:
        item = self.samples[idx]
        # PIL로 이미지를 열고 RGB 3채널로 통일
        try:
            image = Image.open(item["path"]).convert("RGB")
        except Exception as e:
            print(f"[WARN] 이미지 로드 실패, 검은 이미지 대체: {item['path']} | {e}")
            image = Image.new("RGB", (self.image_size, self.image_size), color=0)
        if self.transform is not None:
            # augmentation / resize / normalize 적용
            image = self.transform(image)
        style_idx = self.style_to_index[item["style"]]
        gender_idx = self.gender_to_index[item["gender"]]
        label_idx = self.label_to_index[item["label"]]
        return image, style_idx, gender_idx, label_idx


class BasicBlock(nn.Module):
    """
    ResNet18 계열에서 주로 사용하는 기본 residual block.
    현재 기본 학습 모델은 ResNet50이라 Bottleneck을 사용하지만,
    코드 확장성을 위해 BasicBlock도 유지합니다.
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
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        # 주 경로(main path)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            # skip 경로 크기(채널/해상도) 맞춤
            identity = self.downsample(x)
        # residual 연결: out + identity
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
        # 1x1: 채널 압축/정리
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 3x3: 실제 공간 패턴 학습(모서리, 질감 등)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # 1x1: 채널 확장(expansion=4)
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
    - forward(): style/gender logits 반환
    - extract_feature_vector(): 추천 시스템에서 쓸 임베딩 벡터 반환
    """

    def __init__(
        self,
        img_channels: int,
        num_layers: int,
        block: Type[nn.Module],
        num_style_classes: int,
        num_gender_classes: int,
        dropout_p: float = 0.0,
    ) -> None:
        super(ResNet, self).__init__()
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError("dropout_p must be in [0.0, 1.0).")
        if num_layers == 18:
            # ResNet18 stage 반복 수: [2, 2, 2, 2]
            layers = [2, 2, 2, 2]
            self.expansion = 1
        elif num_layers == 50:
            # ResNet50 stage 반복 수: [3, 4, 6, 3]
            layers = [3, 4, 6, 3]
            self.expansion = 4
        else:
            raise ValueError("num_layers must be one of [18, 50].")

        self.in_channels = 64
        # 7x7, stride=2: 고해상도 입력에서 초반 연산량을 줄이는 기본 stem
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 공통 feature vector를 기반으로 style/gender를 각각 예측하는 2-Head 구조
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()
        feature_dim = 512 * self.expansion
        self.style_fc = nn.Linear(feature_dim, num_style_classes)
        self.gender_fc = nn.Linear(feature_dim, num_gender_classes)
        self.initialize_weights()

    def _make_layer(
        self,
        block: Type[nn.Module],
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        # stride!=1이면 해상도 변화, 채널 불일치면 채널 변화.
        # 이때 skip 경로도 동일한 shape으로 맞춰줘야 out + identity가 가능합니다.
        if stride != 1 or self.in_channels != out_channels * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
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
    
    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_backbone(self, x: Tensor) -> Tensor:
        # 분류 헤드를 제외한 feature extractor(backbone) 부분
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_head(self, feature_map: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # feature map -> (style logits, gender logits, 추천용 feature vector)
        pooled = self.avgpool(feature_map)
        feature_vector = torch.flatten(pooled, 1)
        dropped = self.dropout(feature_vector)
        style_logits = self.style_fc(dropped)
        gender_logits = self.gender_fc(dropped)
        return style_logits, gender_logits, feature_vector

    def extract_feature_vector(self, x: Tensor, normalize: bool = True) -> Tensor:
        # 추천 시스템에서 사용할 임베딩 추출 함수
        feature_map = self.forward_backbone(x)
        _, _, feature_vector = self.forward_head(feature_map)
        if normalize:
            # cosine similarity 사용 시 L2 정규화가 일반적으로 유리
            feature_vector = F.normalize(feature_vector, p=2, dim=1)
        return feature_vector

    def forward(
        self,
        x: Tensor,
        return_features: bool = False,
        return_feature_map: bool = False,
    ):
        feature_map = self.forward_backbone(x)
        style_logits, gender_logits, feature_vector = self.forward_head(feature_map)
        if return_features and return_feature_map:
            return style_logits, gender_logits, feature_vector, feature_map
        if return_features:
            return style_logits, gender_logits, feature_vector
        return style_logits, gender_logits


def resnet50(
    img_channels: int,
    num_style_classes: int,
    num_gender_classes: int,
    dropout_p: float = 0.0,
) -> ResNet:
    """
    편의 함수: ResNet50 인스턴스를 생성.
    """
    return ResNet(
        img_channels,
        50,
        Bottleneck,
        num_style_classes=num_style_classes,
        num_gender_classes=num_gender_classes,
        dropout_p=dropout_p,
    )

def resnet18(
    img_channels: int,
    num_style_classes: int,
    num_gender_classes: int,
    dropout_p: float = 0.0,
) -> ResNet:
    return ResNet(
        img_channels,
        18,
        BasicBlock,
        num_style_classes=num_style_classes,
        num_gender_classes=num_gender_classes,
        dropout_p=dropout_p,
    )

def create_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    학습/검증 전처리 파이프라인 구성.
    - train: augmentation 포함
    - val: resize + normalize 중심
    입력 원본이 3000x4000(3:4)에 가깝다는 가정으로 비율 유지 resize를 적용합니다.
    """
    target_h = image_size
    target_w = int(round(image_size * (4 / 3)))

    train_transform = transforms.Compose([
    transforms.Resize((target_h, target_w)),
    transforms.RandomHorizontalFlip(p=0.5),
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
    target_to_index: Dict[str, int],
    target_key: str,
    device: torch.device,
) -> torch.Tensor:
    """
    클래스 불균형 완화를 위한 손실 가중치 계산.
    샘플이 적은 클래스일수록 더 큰 가중치가 부여됩니다.
    """
    class_counts = np.zeros(len(target_to_index), dtype=np.float32)
    for rec in train_records:
        class_counts[target_to_index[rec[target_key]]] += 1.0

    # 샘플 수가 적은 클래스의 손실을 조금 더 크게 주기 위한 역빈도 가중치
    weights = class_counts.sum() / (len(class_counts) * np.clip(class_counts, 1.0, None))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    style_criterion: nn.Module,
    gender_criterion: nn.Module,
    device: torch.device,
    combined_index_lookup: Dict[Tuple[int, int], int],
    unknown_combined_index: int,
    gender_loss_weight: float,
    log_prefix: str = "",
    log_interval: int = 0,
    data_wait_warn_sec: float = 10.0,
) -> Dict[str, object]:
    """
    지정한 데이터셋(loader)에서 멀티태스크 loss/accuracy를 계산합니다.
    반환되는 labels/preds는 style/gender/combined 지표 계산에 사용됩니다.
    """
    model.eval()
    running_total_loss = 0.0
    running_style_loss = 0.0
    running_gender_loss = 0.0
    all_style_preds: List[int] = []
    all_style_labels: List[int] = []
    all_gender_preds: List[int] = []
    all_gender_labels: List[int] = []
    all_combined_preds: List[int] = []
    all_combined_labels: List[int] = []
    style_correct = 0
    gender_correct = 0
    combined_correct = 0
    total = 0

    eval_prev_end = time.time()
    with torch.no_grad():
        for batch_idx, (images, style_labels, gender_labels, combined_labels) in enumerate(
            loader, start=1
        ):
            batch_fetch_done = time.time()
            data_wait = batch_fetch_done - eval_prev_end
            step_start = time.time()
            images = images.to(device)
            style_labels = style_labels.to(device)
            gender_labels = gender_labels.to(device)
            combined_labels = combined_labels.to(device)

            style_outputs, gender_outputs = model(images)
            style_loss = style_criterion(style_outputs, style_labels)
            gender_loss = gender_criterion(gender_outputs, gender_labels)
            loss = style_loss + (gender_loss_weight * gender_loss)
            running_total_loss += loss.item()
            running_style_loss += style_loss.item()
            running_gender_loss += gender_loss.item()

            _, style_preds = torch.max(style_outputs, 1)
            _, gender_preds = torch.max(gender_outputs, 1)
            total += style_labels.size(0)
            style_correct += (style_preds == style_labels).sum().item()
            gender_correct += (gender_preds == gender_labels).sum().item()
            combined_correct += (
                (style_preds == style_labels) & (gender_preds == gender_labels)
            ).sum().item()

            all_style_preds.extend(style_preds.cpu().tolist())
            all_style_labels.extend(style_labels.cpu().tolist())
            all_gender_preds.extend(gender_preds.cpu().tolist())
            all_gender_labels.extend(gender_labels.cpu().tolist())
            all_combined_labels.extend(combined_labels.cpu().tolist())
            for s_pred, g_pred in zip(
                style_preds.cpu().tolist(),
                gender_preds.cpu().tolist(),
            ):
                pred_idx = combined_index_lookup.get(
                    (s_pred, g_pred),
                    unknown_combined_index,
                )
                all_combined_preds.append(pred_idx)

            if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == len(loader)):
                step_time = time.time() - step_start
                warn_text = ""
                if data_wait > data_wait_warn_sec:
                    warn_text = f" | [WARN] data_wait>{data_wait_warn_sec:.1f}s"
                print(
                    f"{log_prefix} batch {batch_idx}/{len(loader)} | "
                    f"loss={loss.item():.4f} (style={style_loss.item():.4f}, "
                    f"gender={gender_loss.item():.4f}) | data_wait={data_wait:.2f}s | "
                    f"step_time={step_time:.2f}s{warn_text}"
                )
            eval_prev_end = time.time()

    avg_total_loss = running_total_loss / max(len(loader), 1)
    avg_style_loss = running_style_loss / max(len(loader), 1)
    avg_gender_loss = running_gender_loss / max(len(loader), 1)
    style_acc_pct = (style_correct / total) * 100 if total > 0 else 0.0
    gender_acc_pct = (gender_correct / total) * 100 if total > 0 else 0.0
    combined_acc_pct = (combined_correct / total) * 100 if total > 0 else 0.0
    return {
        "total_loss": avg_total_loss,
        "style_loss": avg_style_loss,
        "gender_loss": avg_gender_loss,
        "style_acc_pct": style_acc_pct,
        "gender_acc_pct": gender_acc_pct,
        "combined_acc_pct": combined_acc_pct,
        "style_labels": all_style_labels,
        "style_preds": all_style_preds,
        "gender_labels": all_gender_labels,
        "gender_preds": all_gender_preds,
        "combined_labels": all_combined_labels,
        "combined_preds": all_combined_preds,
    }


def validate_label_coverage(train_records: List[Dict[str, str]]) -> Tuple[int, int, int]:
    """
    멀티태스크(style/gender) + 결합 라벨 지표 계산을 위한 기본 라벨 커버리지 검증.
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


def run_single_training(
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, object]:
    """
    멀티태스크 모델(shared backbone + style/gender head) 학습 파이프라인.
    """
    if not 0.0 <= args.dropout_p < 1.0:
        raise ValueError("--dropout-p must be in [0.0, 1.0).")
    if args.gender_loss_weight < 0.0:
        raise ValueError("--gender-loss-weight must be >= 0.0.")

    # STEP 0) 재현성 설정 + 출력 폴더 준비
    seed_everything(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # STEP 1) 파일 스캔 및 라벨 파싱
    train_records, train_invalid = collect_records(args.train_dir)
    print_label_inventory(
        train_records,
        "Raw train_dir label inventory (before 7:3 split)",
    )

    # 요구사항: train_dir 전체를 label 기준 7:3으로 train/valid 자동 분할
    train_records, val_records = split_train_val_records(
        train_records, val_ratio=0.3, seed=args.seed
    )
    print_label_inventory(
        train_records,
        "Train split label inventory (after 7:3 split)",
    )
    print_label_inventory(
        val_records,
        "Validation split label inventory (after 7:3 split)",
    )
    val_invalid: List[str] = []

    # 요구사항: val_dir는 test 셋 경로로 사용
    if args.val_dir:
        test_records, test_invalid = collect_records(args.val_dir)
        print_label_inventory(
            test_records,
            "Test label inventory (loaded from --val-dir)",
        )
    else:
        test_records = []
        test_invalid = []

    if len(train_records) == 0 or len(val_records) == 0:
        raise ValueError(
            "Train/validation records are empty. "
            "Check directory and filename format."
        )
    if args.val_dir and len(test_records) == 0:
        raise ValueError("Test records are empty. Check test directory and filename format.")

    # STEP 2) 라벨 커버리지 검증
    actual_style_count, actual_gender_count, actual_label_count = validate_label_coverage(train_records)
    print(
        f"[Step 2] Validate labels: style_count={actual_style_count}, "
        f"gender_count={actual_gender_count}, combined_class_count={actual_label_count}"
    )

    # STEP 3) style / gender / combined 라벨 맵 생성
    style_list = sorted({r["style"] for r in train_records})
    gender_list = sorted({r["gender"] for r in train_records})
    label_list = sorted({r["label"] for r in train_records})

    style_to_index = {style: idx for idx, style in enumerate(style_list)}
    gender_to_index = {gender: idx for idx, gender in enumerate(gender_list)}
    label_to_index = {label: idx for idx, label in enumerate(label_list)}

    index_to_style = {idx: style for style, idx in style_to_index.items()}
    index_to_gender = {idx: gender for gender, idx in gender_to_index.items()}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    print("[Step 3] Build style label mapping")
    for idx in range(len(style_list)):
        print(f"  style[{idx}] = {index_to_style[idx]}")
    print("[Step 3] Build gender label mapping")
    for idx in range(len(gender_list)):
        print(f"  gender[{idx}] = {index_to_gender[idx]}")
    print("[Step 3] Build combined label mapping")
    for idx in range(len(label_list)):
        print(f"  class[{idx}] = {index_to_label[idx]}")

    # STEP 4) train에 없는 style/gender/combined 라벨이 val에 있으면 제거
    filtered_val_records = [
        r
        for r in val_records
        if r["style"] in style_to_index
        and r["gender"] in gender_to_index
        and r["label"] in label_to_index
    ]
    dropped_val = len(val_records) - len(filtered_val_records)
    val_records = filtered_val_records
    if len(val_records) == 0:
        raise ValueError("Validation records became empty after label filtering.")
    print("[Step 4] Validate val labels")
    print_label_inventory(
        val_records,
        "Validation labels actually used for training",
    )

    # STEP 4-1) train에 없는 style/gender/combined 라벨이 test에 있으면 제거
    filtered_test_records = [
        r
        for r in test_records
        if r["style"] in style_to_index
        and r["gender"] in gender_to_index
        and r["label"] in label_to_index
    ]
    dropped_test = len(test_records) - len(filtered_test_records)
    test_records = filtered_test_records
    if args.val_dir and len(test_records) == 0:
        raise ValueError("Test records became empty after label filtering.")
    if len(test_records) > 0:
        print_label_inventory(
            test_records,
            "Test labels actually used for final evaluation",
        )

    # STEP 5) Dataset / DataLoader 생성
    train_transform, val_transform = create_transforms(args.image_size)
    train_dataset = FashionStyleDataset(
        train_records,
        style_to_index=style_to_index,
        gender_to_index=gender_to_index,
        label_to_index=label_to_index,
        transform=train_transform,
        image_size=args.image_size,
    )
    val_dataset = FashionStyleDataset(
        val_records,
        style_to_index=style_to_index,
        gender_to_index=gender_to_index,
        label_to_index=label_to_index,
        transform=val_transform,
        image_size=args.image_size,
    )
    test_dataset = (
        FashionStyleDataset(
            test_records,
            style_to_index=style_to_index,
            gender_to_index=gender_to_index,
            label_to_index=label_to_index,
            transform=val_transform,
            image_size=args.image_size,
        )
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
    model = resnet50(
        img_channels=3,
        num_style_classes=len(style_list),
        num_gender_classes=len(gender_list),
        dropout_p=args.dropout_p,
    ).to(device)
    feature_dim = int(model.style_fc.in_features)

    style_weights = (
        build_class_weights(train_records, style_to_index, "style", device)
        if args.use_class_weights
        else None
    )
    gender_weights = (
        build_class_weights(train_records, gender_to_index, "gender", device)
        if args.use_class_weights
        else None
    )
    style_criterion = nn.CrossEntropyLoss(weight=style_weights)
    gender_criterion = nn.CrossEntropyLoss(weight=gender_weights)

    combined_index_lookup: Dict[Tuple[int, int], int] = {}
    for rec in train_records:
        key = (style_to_index[rec["style"]], gender_to_index[rec["gender"]])
        combined_index_lookup[key] = label_to_index[rec["label"]]
    unknown_combined_index = len(label_list)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
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
    print(f"\n[Step 6] Reset ResNet50 Model. Device is {device}")

    print("\n[Step 7] === Start Multi-Task Training (Style + Gender Heads) ===")
    print(
        f"Num style classes: {len(style_list)}, Num gender classes: {len(gender_list)}, "
        f"Num combined classes: {len(label_list)}, Feature dim: {feature_dim}"
    )
    print(
        f"train_samples={len(train_dataset)}, val_samples={len(val_dataset)}, "
        f"train_batches={len(train_loader)}, val_batches={len(val_loader)}, "
        f"batch_size={args.batch_size}, num_workers={args.num_workers}, "
        f"train_val_split=7:3(label-stratified), dropout_p={args.dropout_p}, "
        f"gender_loss_weight={args.gender_loss_weight}"
    )
    if test_loader is not None:
        print(
            f"test_samples={len(test_dataset)}, test_batches={len(test_loader)} "
            f"(final evaluation split from --val-dir)"
        )
    else:
        print("[INFO] --val-dir(=test 경로)가 없어 final metrics는 validation split 기준으로 계산합니다.")
    if len(train_dataset) > 0:
        print(f"example_train_file={train_dataset.samples[0]['path']}")

    # STEP 7) Epoch 학습 루프 + Early Stopping
    for epoch in range(args.num_epochs):
        model.train()
        epoch_start = time.time()
        train_running_total_loss = 0.0
        train_running_style_loss = 0.0
        train_running_gender_loss = 0.0
        train_style_correct = 0
        train_gender_correct = 0
        train_combined_correct = 0
        train_total = 0
        prev_batch_end = time.time()

        print(f"epoch {epoch + 1}/{args.num_epochs} started (waiting first batch...)")

        for batch_idx, (images, style_labels, gender_labels, _) in enumerate(train_loader, start=1):
            batch_fetch_done = time.time()
            data_wait = batch_fetch_done - prev_batch_end
            step_start = time.time()
            images = images.to(device)
            style_labels = style_labels.to(device)
            gender_labels = gender_labels.to(device)

            optimizer.zero_grad()
            style_outputs, gender_outputs = model(images)
            style_loss = style_criterion(style_outputs, style_labels)
            gender_loss = gender_criterion(gender_outputs, gender_labels)
            loss = style_loss + (args.gender_loss_weight * gender_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_running_total_loss += loss.item()
            train_running_style_loss += style_loss.item()
            train_running_gender_loss += gender_loss.item()
            _, style_preds = torch.max(style_outputs, 1)
            _, gender_preds = torch.max(gender_outputs, 1)
            train_total += style_labels.size(0)
            train_style_correct += (style_preds == style_labels).sum().item()
            train_gender_correct += (gender_preds == gender_labels).sum().item()
            train_combined_correct += (
                (style_preds == style_labels) & (gender_preds == gender_labels)
            ).sum().item()

            if args.log_interval > 0 and (batch_idx % args.log_interval == 0 or batch_idx == len(train_loader)):
                step_time = time.time() - step_start
                current_lr = optimizer.param_groups[0]["lr"]
                samples_per_sec = style_labels.size(0) / max(step_time, 1e-8)
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
                    f"loss={loss.item():.4f} (style={style_loss.item():.4f}, "
                    f"gender={gender_loss.item():.4f}) | lr={current_lr:.6f} | "
                    f"data_wait={data_wait:.2f}s | step_time={step_time:.2f}s | "
                    f"img_per_sec={samples_per_sec:.1f}{gpu_mem_text}{warn_text}"
                )
            prev_batch_end = time.time()

        train_total_loss = train_running_total_loss / max(len(train_loader), 1)
        train_style_loss = train_running_style_loss / max(len(train_loader), 1)
        train_gender_loss = train_running_gender_loss / max(len(train_loader), 1)
        train_style_acc = (train_style_correct / train_total) * 100 if train_total > 0 else 0.0
        train_gender_acc = (train_gender_correct / train_total) * 100 if train_total > 0 else 0.0
        train_combined_acc = (train_combined_correct / train_total) * 100 if train_total > 0 else 0.0

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            style_criterion=style_criterion,
            gender_criterion=gender_criterion,
            device=device,
            combined_index_lookup=combined_index_lookup,
            unknown_combined_index=unknown_combined_index,
            gender_loss_weight=args.gender_loss_weight,
            log_prefix=f"val epoch {epoch + 1}/{args.num_epochs}",
            log_interval=args.log_interval,
            data_wait_warn_sec=args.data_wait_warn_sec,
        )
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch + 1}/{args.num_epochs}] "
            f"Train Loss(total/style/gender): "
            f"{train_total_loss:.4f}/{train_style_loss:.4f}/{train_gender_loss:.4f} | "
            f"Train Acc(style/gender/combined): "
            f"{train_style_acc:.2f}%/{train_gender_acc:.2f}%/{train_combined_acc:.2f}% | "
            f"Val Loss(total/style/gender): "
            f"{val_metrics['total_loss']:.4f}/{val_metrics['style_loss']:.4f}/{val_metrics['gender_loss']:.4f} | "
            f"Val Acc(style/gender/combined): "
            f"{val_metrics['style_acc_pct']:.2f}%/{val_metrics['gender_acc_pct']:.2f}%/{val_metrics['combined_acc_pct']:.2f}% | "
            f"epoch_time={epoch_time:.2f}s"
        )

        scheduler.step(float(val_metrics["total_loss"]))
        if float(val_metrics["total_loss"]) < best_val_loss:
            best_val_loss = float(val_metrics["total_loss"])
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model_state.pth")
            print(
                f"best model updated at epoch {epoch + 1} "
                f"(val_total_loss={val_metrics['total_loss']:.4f})"
            )
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
    final_eval_metrics = evaluate(
        model=model,
        loader=final_eval_loader,
        style_criterion=style_criterion,
        gender_criterion=gender_criterion,
        device=device,
        combined_index_lookup=combined_index_lookup,
        unknown_combined_index=unknown_combined_index,
        gender_loss_weight=args.gender_loss_weight,
        log_prefix=f"[{final_eval_split}] final",
        log_interval=0,
        data_wait_warn_sec=args.data_wait_warn_sec,
    )

    # STEP 9) 최종 평가 지표 계산
    style_accuracy = accuracy_score(
        final_eval_metrics["style_labels"],
        final_eval_metrics["style_preds"],
    )
    style_precision_macro, style_recall_macro, style_f1_macro, _ = precision_recall_fscore_support(
        final_eval_metrics["style_labels"],
        final_eval_metrics["style_preds"],
        labels=list(range(len(style_list))),
        average="macro",
        zero_division=0,
    )
    style_precision_weighted, style_recall_weighted, style_f1_weighted, _ = precision_recall_fscore_support(
        final_eval_metrics["style_labels"],
        final_eval_metrics["style_preds"],
        labels=list(range(len(style_list))),
        average="weighted",
        zero_division=0,
    )

    gender_accuracy = accuracy_score(
        final_eval_metrics["gender_labels"],
        final_eval_metrics["gender_preds"],
    )
    gender_precision_macro, gender_recall_macro, gender_f1_macro, _ = precision_recall_fscore_support(
        final_eval_metrics["gender_labels"],
        final_eval_metrics["gender_preds"],
        labels=list(range(len(gender_list))),
        average="macro",
        zero_division=0,
    )
    gender_precision_weighted, gender_recall_weighted, gender_f1_weighted, _ = precision_recall_fscore_support(
        final_eval_metrics["gender_labels"],
        final_eval_metrics["gender_preds"],
        labels=list(range(len(gender_list))),
        average="weighted",
        zero_division=0,
    )

    combined_metric_labels = list(range(len(label_list) + 1))
    combined_target_names = [
        index_to_label[i] for i in range(len(label_list))
    ] + ["__unknown_pair__"]
    combined_accuracy = accuracy_score(
        final_eval_metrics["combined_labels"],
        final_eval_metrics["combined_preds"],
    )
    combined_precision_macro, combined_recall_macro, combined_f1_macro, _ = precision_recall_fscore_support(
        final_eval_metrics["combined_labels"],
        final_eval_metrics["combined_preds"],
        labels=combined_metric_labels,
        average="macro",
        zero_division=0,
    )
    combined_precision_weighted, combined_recall_weighted, combined_f1_weighted, _ = precision_recall_fscore_support(
        final_eval_metrics["combined_labels"],
        final_eval_metrics["combined_preds"],
        labels=combined_metric_labels,
        average="weighted",
        zero_division=0,
    )

    # STEP 10) confusion matrix / classification report 저장
    style_cm = confusion_matrix(
        final_eval_metrics["style_labels"],
        final_eval_metrics["style_preds"],
        labels=list(range(len(style_list))),
    )
    style_cm_df = pd.DataFrame(
        style_cm,
        index=[f"true_{index_to_style[i]}" for i in range(len(style_list))],
        columns=[f"pred_{index_to_style[i]}" for i in range(len(style_list))],
    )
    style_cm_df.to_csv(output_dir / "style_confusion_matrix.csv", encoding="utf-8-sig")
    style_cm_df.to_csv(
        output_dir / f"{final_eval_split}_style_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    gender_cm = confusion_matrix(
        final_eval_metrics["gender_labels"],
        final_eval_metrics["gender_preds"],
        labels=list(range(len(gender_list))),
    )
    gender_cm_df = pd.DataFrame(
        gender_cm,
        index=[f"true_{index_to_gender[i]}" for i in range(len(gender_list))],
        columns=[f"pred_{index_to_gender[i]}" for i in range(len(gender_list))],
    )
    gender_cm_df.to_csv(output_dir / "gender_confusion_matrix.csv", encoding="utf-8-sig")
    gender_cm_df.to_csv(
        output_dir / f"{final_eval_split}_gender_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    combined_cm = confusion_matrix(
        final_eval_metrics["combined_labels"],
        final_eval_metrics["combined_preds"],
        labels=combined_metric_labels,
    )
    combined_cm_df = pd.DataFrame(
        combined_cm,
        index=[f"true_{name}" for name in combined_target_names],
        columns=[f"pred_{name}" for name in combined_target_names],
    )
    combined_cm_df.to_csv(output_dir / "combined_confusion_matrix.csv", encoding="utf-8-sig")
    combined_cm_df.to_csv(
        output_dir / f"{final_eval_split}_combined_confusion_matrix.csv",
        encoding="utf-8-sig",
    )
    # 기존 파일명 호환성 유지
    combined_cm_df.to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    combined_cm_df.to_csv(
        output_dir / f"{final_eval_split}_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    style_report = classification_report(
        final_eval_metrics["style_labels"],
        final_eval_metrics["style_preds"],
        labels=list(range(len(style_list))),
        target_names=[index_to_style[i] for i in range(len(style_list))],
        digits=4,
        zero_division=0,
    )
    with open(output_dir / "style_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(style_report)
    with open(
        output_dir / f"{final_eval_split}_style_classification_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(style_report)

    gender_report = classification_report(
        final_eval_metrics["gender_labels"],
        final_eval_metrics["gender_preds"],
        labels=list(range(len(gender_list))),
        target_names=[index_to_gender[i] for i in range(len(gender_list))],
        digits=4,
        zero_division=0,
    )
    with open(output_dir / "gender_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(gender_report)
    with open(
        output_dir / f"{final_eval_split}_gender_classification_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(gender_report)

    combined_report = classification_report(
        final_eval_metrics["combined_labels"],
        final_eval_metrics["combined_preds"],
        labels=combined_metric_labels,
        target_names=combined_target_names,
        digits=4,
        zero_division=0,
    )
    with open(output_dir / "combined_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(combined_report)
    with open(
        output_dir / f"{final_eval_split}_combined_classification_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(combined_report)
    # 기존 파일명 호환성 유지
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(combined_report)
    with open(output_dir / f"{final_eval_split}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(combined_report)

    # STEP 11) 재현/분석용 메타데이터 저장
    save_label_distribution(train_records, output_dir / "train_label_distribution.csv")
    save_label_distribution(val_records, output_dir / "val_label_distribution.csv")
    if len(test_records) > 0:
        save_label_distribution(test_records, output_dir / "test_label_distribution.csv")
    pd.DataFrame(train_records).to_csv(output_dir / "train_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(val_records).to_csv(output_dir / "val_manifest.csv", index=False, encoding="utf-8-sig")
    if len(test_records) > 0:
        pd.DataFrame(test_records).to_csv(output_dir / "test_manifest.csv", index=False, encoding="utf-8-sig")

    with open(output_dir / "style_to_index.json", "w", encoding="utf-8") as f:
        json.dump(style_to_index, f, ensure_ascii=False, indent=2)
    with open(output_dir / "index_to_style.json", "w", encoding="utf-8") as f:
        json.dump(index_to_style, f, ensure_ascii=False, indent=2)
    with open(output_dir / "gender_to_index.json", "w", encoding="utf-8") as f:
        json.dump(gender_to_index, f, ensure_ascii=False, indent=2)
    with open(output_dir / "index_to_gender.json", "w", encoding="utf-8") as f:
        json.dump(index_to_gender, f, ensure_ascii=False, indent=2)
    with open(output_dir / "label_to_index.json", "w", encoding="utf-8") as f:
        json.dump(label_to_index, f, ensure_ascii=False, indent=2)
    with open(output_dir / "index_to_label.json", "w", encoding="utf-8") as f:
        json.dump(index_to_label, f, ensure_ascii=False, indent=2)

    unknown_pair_pred_count = int(
        np.sum(np.array(final_eval_metrics["combined_preds"]) == unknown_combined_index)
    )

    # STEP 12) 실행 요약 저장
    run_summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_eval_split": final_eval_split,
        "final_eval_total_loss": final_eval_metrics["total_loss"],
        "final_eval_style_loss": final_eval_metrics["style_loss"],
        "final_eval_gender_loss": final_eval_metrics["gender_loss"],
        "style_accuracy": style_accuracy,
        "style_accuracy_percent": style_accuracy * 100.0,
        "style_precision_macro": style_precision_macro,
        "style_recall_macro": style_recall_macro,
        "style_f1_macro": style_f1_macro,
        "style_precision_weighted": style_precision_weighted,
        "style_recall_weighted": style_recall_weighted,
        "style_f1_weighted": style_f1_weighted,
        "gender_accuracy": gender_accuracy,
        "gender_accuracy_percent": gender_accuracy * 100.0,
        "gender_precision_macro": gender_precision_macro,
        "gender_recall_macro": gender_recall_macro,
        "gender_f1_macro": gender_f1_macro,
        "gender_precision_weighted": gender_precision_weighted,
        "gender_recall_weighted": gender_recall_weighted,
        "gender_f1_weighted": gender_f1_weighted,
        "combined_accuracy": combined_accuracy,
        "combined_accuracy_percent": combined_accuracy * 100.0,
        "accuracy": combined_accuracy,
        "accuracy_percent": combined_accuracy * 100.0,
        "final_eval_accuracy_percent": combined_accuracy * 100.0,
        "combined_precision_macro": combined_precision_macro,
        "combined_recall_macro": combined_recall_macro,
        "combined_f1_macro": combined_f1_macro,
        "combined_precision_weighted": combined_precision_weighted,
        "combined_recall_weighted": combined_recall_weighted,
        "combined_f1_weighted": combined_f1_weighted,
        "precision_macro": combined_precision_macro,
        "recall_macro": combined_recall_macro,
        "f1_macro": combined_f1_macro,
        "precision_weighted": combined_precision_weighted,
        "recall_weighted": combined_recall_weighted,
        "f1_weighted": combined_f1_weighted,
        "combined_unknown_pair_pred_count": unknown_pair_pred_count,
        "num_train_images": len(train_records),
        "num_val_images": len(val_records),
        "num_test_images": len(test_records),
        "num_style_classes": len(style_list),
        "num_gender_classes": len(gender_list),
        "num_combined_classes": len(label_list),
        "num_classes": len(label_list),
        "feature_dim": feature_dim,
        "actual_style_count": actual_style_count,
        "actual_gender_count": actual_gender_count,
        "actual_label_count": actual_label_count,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "dropout_p": args.dropout_p,
        "gender_loss_weight": args.gender_loss_weight,
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
    torch.save(model, output_dir / "last_model_full.pth")

    print(f"\n=== Final Evaluation ({final_eval_split.upper()}) ===")
    print(f"Best Epoch: {best_epoch}")
    print(
        f"Final Loss(total/style/gender): "
        f"{final_eval_metrics['total_loss']:.4f}/"
        f"{final_eval_metrics['style_loss']:.4f}/"
        f"{final_eval_metrics['gender_loss']:.4f}"
    )
    print(
        f"Final Acc(style/gender/combined): "
        f"{style_accuracy * 100:.2f}%/"
        f"{gender_accuracy * 100:.2f}%/"
        f"{combined_accuracy * 100:.2f}%"
    )
    print(
        f"Style Macro    - Precision: {style_precision_macro:.4f}, "
        f"Recall: {style_recall_macro:.4f}, F1: {style_f1_macro:.4f}"
    )
    print(
        f"Combined Macro - Precision: {combined_precision_macro:.4f}, "
        f"Recall: {combined_recall_macro:.4f}, F1: {combined_f1_macro:.4f}"
    )
    print(f"Output directory: {output_dir.resolve()}")

    return run_summary


def run_training(args: argparse.Namespace) -> None:
    """
    멀티태스크 모델 학습 실행.
    """
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    summary = run_single_training(
        args=args,
        output_dir=base_output_dir,
    )
    with open(base_output_dir / "multi_task_run_summary.json", "w", encoding="utf-8") as f:
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
        help="Test image directory. (기존 val-dir 인자 재사용)",
    )
    parser.add_argument("--image-size", type=int, default=224, help="입력 이미지 크기")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size (OOM 시 8/4 권장)")
    parser.add_argument("--num-epochs", type=int, default=30, help="학습 epoch")
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument(
        "--dropout-p",
        type=float,
        default=0.3,
        help="분류 헤드 드롭아웃 확률 [0.0, 1.0).",
    )
    parser.add_argument(
        "--gender-loss-weight",
        type=float,
        default=0.3,
        help="총 손실 = style_loss + gender_loss_weight * gender_loss",
    )
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
