import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None


@dataclass
class CatalogData:
    embeddings: np.ndarray
    metadata: pd.DataFrame
    key_column: str
    style_column: str
    gender_column: str
    path_column: Optional[str]
    key_to_indices: Dict[str, List[int]]
    style_prototypes: Dict[str, np.ndarray]
    faiss_index: Optional["faiss.Index"]


@dataclass
class LoadedResources:
    catalog: CatalogData
    model: nn.Module
    device: torch.device
    infer_transform: transforms.Compose
    survey_profile: Optional["SurveyProfile"]


@dataclass
class SurveyProfile:
    style_scores_global: Dict[str, float]
    style_scores_by_gender: Dict[str, Dict[str, float]]
    prior_vector_global: Optional[np.ndarray]
    prior_vector_by_gender: Dict[str, np.ndarray]
    source_rows: int

    def get_prior_vector(self, gender_filter: Optional[str]) -> Optional[np.ndarray]:
        if gender_filter in {"male", "female"} and gender_filter in self.prior_vector_by_gender:
            return self.prior_vector_by_gender[gender_filter]
        return self.prior_vector_global


class ResourceFactory:
    DEFAULT_DROPOUT = 0.2

    @staticmethod
    def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1 if x.ndim == 2 else 0, keepdims=True)
        return x / np.clip(norm, eps, None)

    @staticmethod
    def parse_style_gender_from_name(file_name: str) -> Tuple[Optional[str], Optional[str]]:
        stem = Path(file_name).stem
        parts = stem.split("_")
        if len(parts) < 5:
            return None, None
        style = parts[3].strip().lower() if parts[3].strip() else None
        gender_token = parts[-1].strip().upper()
        if gender_token == "W":
            gender = "female"
        elif gender_token == "M":
            gender = "male"
        else:
            gender = None
        return style, gender

    @staticmethod
    def normalize_gender_token(token: Optional[str]) -> Optional[str]:
        if token is None:
            return None
        t = str(token).strip().lower()
        if t in {"male", "m", "man", "men"}:
            return "male"
        if t in {"female", "f", "w", "woman", "women"}:
            return "female"
        return None

    @classmethod
    def split_style_and_gender_label(cls, style_label: object) -> Tuple[Optional[str], Optional[str]]:
        raw = str(style_label).strip().lower()
        if not raw or raw == "nan":
            return None, None
        head, sep, tail = raw.rpartition("_")
        if sep:
            parsed_gender = cls.normalize_gender_token(tail)
            if parsed_gender is not None and head.strip():
                return head.strip(), parsed_gender
        return raw, None

    @classmethod
    def make_style_gender_key(cls, style: object, gender: object) -> str:
        style_norm = str(style).strip().lower()
        if not style_norm:
            style_norm = "unknown"
        gender_norm = cls.normalize_gender_token(str(gender))
        if gender_norm in {"male", "female"}:
            return f"{style_norm}_{gender_norm}"
        return style_norm

    @classmethod
    def get_style_prototype(
        cls,
        style_key: str,
        style_prototypes: Dict[str, np.ndarray],
        gender_filter: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        base = str(style_key).strip().lower()
        if not base or base == "unknown":
            return None

        gender_norm = cls.normalize_gender_token(gender_filter)
        if gender_norm in {"male", "female"}:
            key = cls.make_style_gender_key(base, gender_norm)
            if key in style_prototypes:
                return style_prototypes[key]

        if base in style_prototypes:
            return style_prototypes[base]

        candidates: List[np.ndarray] = []
        for g in ["male", "female"]:
            key = cls.make_style_gender_key(base, g)
            if key in style_prototypes:
                candidates.append(style_prototypes[key])
        if len(candidates) == 0:
            return None
        if len(candidates) == 1:
            return candidates[0]
        avg = np.mean(np.stack(candidates, axis=0), axis=0).astype(np.float32)
        return cls.l2_normalize(avg)

    @staticmethod
    def split_csv_list(value: object) -> List[str]: 
        if value is None:
            return []
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return []
        text = text.strip("[]")
        tokens = [token.strip().strip("'").strip('"') for token in text.split(",")]
        return [token for token in tokens if token]

    @staticmethod
    def detect_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
        lower_to_real = {col.lower(): col for col in df.columns}
        for cand in candidates:
            if cand.lower() in lower_to_real:
                return lower_to_real[cand.lower()]
        return None

    @staticmethod
    def detect_survey_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        like_cols: List[str] = []
        dislike_cols: List[str] = []

        # Keep literals ASCII-safe using Unicode escapes.
        kr_like = "\uC120\uD638"
        kr_dislike = "\uBE44\uC120\uD638"

        exact_like = {
            "train \uC120\uD638",
            "valid \uC120\uD638",
            "train_like",
            "valid_like",
            "train_prefer",
            "valid_prefer",
        }
        exact_dislike = {
            "train \uBE44\uC120\uD638",
            "valid \uBE44\uC120\uD638",
            "train_dislike",
            "valid_dislike",
            "train_hate",
            "valid_hate",
        }

        for col in df.columns:
            col_strip = str(col).strip()
            col_lower = col_strip.lower()
            col_compact = col_lower.replace(" ", "")

            if col_strip in exact_dislike or col_lower in exact_dislike:
                dislike_cols.append(col)
                continue
            if col_strip in exact_like or col_lower in exact_like:
                like_cols.append(col)
                continue

            # Fallback matching for mixed naming conventions.
            if (kr_dislike in col_strip) or ("dislike" in col_compact) or ("hate" in col_compact):
                dislike_cols.append(col)
                continue
            if (kr_like in col_strip) or (
                ("like" in col_compact or "prefer" in col_compact) and "dislike" not in col_compact
            ):
                like_cols.append(col)

        return like_cols, dislike_cols

    @classmethod
    def build_prior_vector_from_scores(
        cls,
        style_scores: Dict[str, float],
        style_prototypes: Dict[str, np.ndarray],
        gender_filter: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        vec: Optional[np.ndarray] = None
        for style, score in style_scores.items():
            proto = cls.get_style_prototype(
                style_key=style,
                style_prototypes=style_prototypes,
                gender_filter=gender_filter,
            )
            if proto is None:
                continue
            contrib = float(score) * proto
            vec = contrib.copy() if vec is None else vec + contrib
        if vec is None:
            return None
        return cls.l2_normalize(vec.astype(np.float32))

    @classmethod
    def build_survey_profile(
        cls,
        survey_csv: Path,
        style_prototypes: Dict[str, np.ndarray],
    ) -> SurveyProfile:
        if not survey_csv.exists():
            raise FileNotFoundError(f"Survey CSV not found: {survey_csv}")

        df = pd.read_csv(survey_csv)
        like_cols, dislike_cols = cls.detect_survey_columns(df)
        if len(like_cols) == 0 and len(dislike_cols) == 0:
            raise ValueError(
                "Survey CSV must include preference/dislike columns. "
                "Examples: train_like / valid_like / train_dislike / valid_dislike"
            )

        global_like: Dict[str, int] = {}
        global_dislike: Dict[str, int] = {}
        by_gender_like: Dict[str, Dict[str, int]] = {"male": {}, "female": {}}
        by_gender_dislike: Dict[str, Dict[str, int]] = {"male": {}, "female": {}}

        def _bump(bucket: Dict[str, int], key: str) -> None:
            bucket[key] = int(bucket.get(key, 0)) + 1

        for _, row in df.iterrows():
            for col in like_cols:
                for file_name in cls.split_csv_list(row[col]):
                    style, gender = cls.parse_style_gender_from_name(file_name)
                    if not style:
                        continue
                    _bump(global_like, style)
                    if gender in {"male", "female"}:
                        _bump(by_gender_like[gender], style)
            for col in dislike_cols:
                for file_name in cls.split_csv_list(row[col]):
                    style, gender = cls.parse_style_gender_from_name(file_name)
                    if not style:
                        continue
                    _bump(global_dislike, style)
                    if gender in {"male", "female"}:
                        _bump(by_gender_dislike[gender], style)

        def _scores_from_counts(like_map: Dict[str, int], dislike_map: Dict[str, int]) -> Dict[str, float]:
            all_styles = set(like_map.keys()) | set(dislike_map.keys())
            scores: Dict[str, float] = {}
            for s in all_styles:
                l = float(like_map.get(s, 0))
                d = float(dislike_map.get(s, 0))
                denom = l + d
                if denom <= 0:
                    continue
                # Range [-1, +1]: positive means globally preferred in survey.
                scores[s] = (l - d) / denom
            return scores

        global_scores = _scores_from_counts(global_like, global_dislike)
        gender_scores = {
            "male": _scores_from_counts(by_gender_like["male"], by_gender_dislike["male"]),
            "female": _scores_from_counts(by_gender_like["female"], by_gender_dislike["female"]),
        }

        prior_global = cls.build_prior_vector_from_scores(
            global_scores,
            style_prototypes,
            gender_filter=None,
        )
        prior_by_gender: Dict[str, np.ndarray] = {}
        for g in ["male", "female"]:
            vec = cls.build_prior_vector_from_scores(
                gender_scores[g],
                style_prototypes,
                gender_filter=g,
            )
            if vec is not None:
                prior_by_gender[g] = vec

        return SurveyProfile(
            style_scores_global=global_scores,
            style_scores_by_gender=gender_scores,
            prior_vector_global=prior_global,
            prior_vector_by_gender=prior_by_gender,
            source_rows=int(len(df)),
        )

    @classmethod
    def load_embeddings(cls, emb_path: Path) -> np.ndarray:
        if not emb_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {emb_path}")

        if emb_path.suffix.lower() == ".npy":
            emb = np.load(emb_path)
        elif emb_path.suffix.lower() == ".npz":
            data = np.load(emb_path)
            key_candidates = ["embeddings", "features", "vectors"]
            key = next((k for k in key_candidates if k in data.files), None)
            if key is None:
                if len(data.files) != 1:
                    raise ValueError(
                        f"NPZ file must contain one array or one of {key_candidates}, got keys={data.files}"
                    )
                key = data.files[0]
            emb = data[key]
        else:
            raise ValueError("Unsupported embedding format. Use .npy or .npz")

        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError(f"Embeddings must be 2D [N, D], got shape={emb.shape}")
        return cls.l2_normalize(emb)

    @classmethod
    def build_style_prototypes(
        cls,
        embeddings: np.ndarray,
        styles: Iterable[str],
    ) -> Dict[str, np.ndarray]:
        groups: Dict[str, List[int]] = {}
        for idx, style in enumerate(styles):
            style_norm = str(style).strip().lower()
            if not style_norm:
                continue
            groups.setdefault(style_norm, []).append(idx)

        prototypes: Dict[str, np.ndarray] = {}
        for style, idxs in groups.items():
            vec = embeddings[idxs].mean(axis=0).astype(np.float32)
            prototypes[style] = cls.l2_normalize(vec)
        return prototypes

    @classmethod
    def load_catalog(
        cls,
        metadata_csv: Path,
        embeddings_path: Path,
        faiss_index_path: Optional[Path] = None,
    ) -> CatalogData:
        if not metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

        metadata = pd.read_csv(metadata_csv)
        embeddings = cls.load_embeddings(embeddings_path)
        if len(metadata) != len(embeddings):
            raise ValueError(
                f"Metadata row count and embedding row count mismatch: {len(metadata)} vs {len(embeddings)}"
            )

        key_column = cls.detect_column(
            metadata,
            [
                "image_name",
                "file_name",
                "filename",
                "name",
                "image_id",
                "basename",
                "path",
                "image_path",
            ],
        )
        if key_column is None:
            raise ValueError(
                "Metadata CSV must include one key column among: "
                "image_name, file_name, filename, name, image_id, basename, path, image_path"
            )

        style_column = cls.detect_column(metadata, ["style", "style_name"])
        gender_column = cls.detect_column(metadata, ["gender", "sex"])
        path_column = cls.detect_column(metadata, ["path", "image_path", "img_path", "full_path"])

        style_values: List[str] = []
        gender_values: List[str] = []
        key_to_indices: Dict[str, List[int]] = {}

        for idx, row in metadata.iterrows():
            key_raw = str(row[key_column]).strip()
            key = Path(key_raw).name.lower()
            key_to_indices.setdefault(key, []).append(idx)

            inferred_style, inferred_gender = cls.parse_style_gender_from_name(key_raw)
            if style_column is None:
                raw_style = inferred_style or "unknown"
            else:
                val = str(row[style_column]).strip().lower()
                raw_style = val if val else (inferred_style or "unknown")

            style_from_label, gender_from_style = cls.split_style_and_gender_label(raw_style)
            style_norm = style_from_label or inferred_style or "unknown"
            style_values.append(style_norm)

            if gender_column is None:
                gender_raw = inferred_gender
            else:
                gender_raw = str(row[gender_column]).strip().lower()
            gender_norm = (
                cls.normalize_gender_token(gender_raw)
                or gender_from_style
                or cls.normalize_gender_token(inferred_gender)
                or "unknown"
            )
            gender_values.append(gender_norm)

        metadata = metadata.copy()
        metadata["__style__"] = style_values
        metadata["__gender__"] = gender_values
        metadata["__style_gender__"] = [
            cls.make_style_gender_key(style, gender)
            for style, gender in zip(metadata["__style__"], metadata["__gender__"])
        ]
        prototypes = cls.build_style_prototypes(embeddings, metadata["__style_gender__"].tolist())

        faiss_index = None
        if faiss_index_path and faiss_index_path.exists() and faiss is not None:
            faiss_index = faiss.read_index(str(faiss_index_path))
            if faiss_index.d != embeddings.shape[1]:
                raise ValueError(
                    f"FAISS index dimension mismatch: index.d={faiss_index.d}, emb_dim={embeddings.shape[1]}"
                )

        return CatalogData(
            embeddings=embeddings,
            metadata=metadata,
            key_column=key_column,
            style_column="__style__",
            gender_column="__gender__",
            path_column=path_column,
            key_to_indices=key_to_indices,
            style_prototypes=prototypes,
            faiss_index=faiss_index,
        )

    @staticmethod
    def load_num_classes(label_map_path: Optional[Path], fallback: int) -> int:
        if label_map_path is None or not label_map_path.exists():
            return fallback
        with open(label_map_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(len(data))


    @staticmethod
    def load_model(
        checkpoint_path: Path,
        num_classes: int,
        device: torch.device,
        dropout: float,
    ) -> nn.Module:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        from model_single_task_learning import resnet50

        loaded = torch.load(checkpoint_path, map_location=device)
        state_dict = loaded
        if isinstance(loaded, dict) and "state_dict" in loaded:
            state_dict = loaded["state_dict"]
        if not isinstance(state_dict, dict):
            raise ValueError(
                "Checkpoint format not supported. Please pass a state_dict-based checkpoint."
            )
        if any(k.startswith("backbone.") for k in state_dict.keys()):
            raise ValueError(
                "Team wrapper checkpoint(backbone./classifier.) is not supported here. "
                "Please use custom ResNet checkpoint keys(conv1/layer*/fc*)."
            )

        model = resnet50(img_channels=3, num_classes=num_classes, dropout_p=dropout)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model

    @staticmethod
    def create_infer_transform(
        image_size: int,
        transform_profile: str = "team",
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> transforms.Compose:
        if transform_profile.lower() == "team":
            resize_size = (2 * image_size, image_size)
        elif transform_profile.lower() == "square":
            resize_size = (image_size, image_size)
        else:
            raise ValueError("transform_profile must be one of ['team', 'square']")

        return transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    @classmethod
    def build_loaded_resources(
        cls,
        metadata_csv: str,
        embeddings_npy: str,
        faiss_index: str,
        model_checkpoint: str,
        label_map_json: str,
        survey_csv: str,
        use_survey_csv_prior: bool,
        num_classes: int,
        device_name: str,
        image_size: int,
        transform_profile: str,
        dropout: float = DEFAULT_DROPOUT,
    ) -> LoadedResources:
        device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
        catalog = cls.load_catalog(
            metadata_csv=Path(metadata_csv),
            embeddings_path=Path(embeddings_npy),
            faiss_index_path=Path(faiss_index) if faiss_index else None,
        )
        classes = cls.load_num_classes(
            Path(label_map_json) if label_map_json else None,
            fallback=num_classes,
        )
        model = cls.load_model(
            checkpoint_path=Path(model_checkpoint),
            num_classes=classes,
            device=device,
            dropout=dropout,
        )
        infer_transform = cls.create_infer_transform(
            image_size=image_size,
            transform_profile=transform_profile,
        )
        survey_profile: Optional[SurveyProfile] = None
        if use_survey_csv_prior and survey_csv.strip():
            survey_profile = cls.build_survey_profile(
                survey_csv=Path(survey_csv),
                style_prototypes=catalog.style_prototypes,
            )
        return LoadedResources(
            catalog=catalog,
            model=model,
            device=device,
            infer_transform=infer_transform,
            survey_profile=survey_profile,
        )


class Recommender:
    def __init__(self, resources: LoadedResources):
        self.resources = resources

    def extract_feature_from_pil(self, image: Image.Image) -> np.ndarray:
        x = self.resources.infer_transform(image.convert("RGB")).unsqueeze(0).to(self.resources.device)
        with torch.inference_mode():
            feat = self.resources.model.extract_feature_vector(x, normalize=True)
        return feat.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def apply_survey_vector_adjustment(
        self,
        query_vector: np.ndarray,
        liked_styles: Sequence[str],
        disliked_styles: Sequence[str],
        gender_filter: str,
        alpha_query: float = 1.0,
        beta_like: float = 0.8,
        gamma_dislike: float = 0.8,
        delta_survey_prior: float = 0.0,
    ) -> np.ndarray:
        vec = alpha_query * query_vector.astype(np.float32)
        for style in liked_styles:
            proto = ResourceFactory.get_style_prototype(
                style_key=style,
                style_prototypes=self.resources.catalog.style_prototypes,
                gender_filter=gender_filter,
            )
            if proto is not None:
                vec += beta_like * proto
        for style in disliked_styles:
            proto = ResourceFactory.get_style_prototype(
                style_key=style,
                style_prototypes=self.resources.catalog.style_prototypes,
                gender_filter=gender_filter,
            )
            if proto is not None:
                vec -= gamma_dislike * proto
        if self.resources.survey_profile is not None and delta_survey_prior != 0.0:
            prior_vec = self.resources.survey_profile.get_prior_vector(gender_filter)
            if prior_vec is not None:
                vec += float(delta_survey_prior) * prior_vec
        return ResourceFactory.l2_normalize(vec)

    def search_catalog(
        self,
        query_vector: np.ndarray,
        top_k: int,
        gender_filter: Optional[str] = None,
        use_faiss_if_possible: bool = True,
    ) -> List[Tuple[int, float]]:
        catalog = self.resources.catalog
        q = query_vector.astype(np.float32)
        metadata = catalog.metadata

        if gender_filter and gender_filter.lower() not in {"all", "any"}:
            gender_norm = gender_filter.lower()
            candidate_idx = np.where(metadata[catalog.gender_column].values == gender_norm)[0]
        else:
            candidate_idx = np.arange(len(metadata))

        if len(candidate_idx) == 0:
            return []

        if (
            use_faiss_if_possible
            and catalog.faiss_index is not None
            and len(candidate_idx) == len(metadata)
        ):
            scores, indices = catalog.faiss_index.search(q[None, :], top_k)
            return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]

        sub_emb = catalog.embeddings[candidate_idx]
        sims = sub_emb @ q
        local_top = np.argsort(-sims)[:top_k]
        return [(int(candidate_idx[i]), float(sims[i])) for i in local_top]

    @staticmethod
    def resolve_image_path(
        row: pd.Series,
        catalog_root: Optional[Path],
        key_column: str,
        path_column: Optional[str],
    ) -> Optional[Path]:
        raw_path = None
        if path_column and path_column in row:
            val = str(row[path_column]).strip()
            if val and val.lower() != "nan":
                raw_path = val
        if raw_path is None:
            raw_path = str(row[key_column]).strip()

        p = Path(raw_path)
        if p.exists():
            return p
        if catalog_root is not None:
            p2 = catalog_root / raw_path
            if p2.exists():
                return p2
            p3 = catalog_root / Path(raw_path).name
            if p3.exists():
                return p3
        return None

    def recommend(
        self,
        image: Image.Image,
        liked_styles: Sequence[str],
        disliked_styles: Sequence[str],
        top_k: int,
        gender_filter: str,
        use_faiss: bool,
        alpha_query: float,
        beta_like: float,
        gamma_dislike: float,
        delta_survey_prior: float,
        catalog_root: Optional[Path],
    ) -> pd.DataFrame:
        query_vec = self.extract_feature_from_pil(image)
        adjusted_vec = self.apply_survey_vector_adjustment(
            query_vector=query_vec,
            liked_styles=liked_styles,
            disliked_styles=disliked_styles,
            gender_filter=gender_filter,
            alpha_query=alpha_query,
            beta_like=beta_like,
            gamma_dislike=gamma_dislike,
            delta_survey_prior=delta_survey_prior,
        )
        results = self.search_catalog(
            query_vector=adjusted_vec,
            top_k=int(top_k),
            gender_filter=gender_filter,
            use_faiss_if_possible=use_faiss,
        )
        if len(results) == 0:
            return pd.DataFrame()

        rows: List[Dict[str, object]] = []
        catalog = self.resources.catalog
        for rank, (idx, score) in enumerate(results, start=1):
            row = catalog.metadata.iloc[idx]
            img_key = str(row[catalog.key_column])
            img_path = self.resolve_image_path(
                row=row,
                catalog_root=catalog_root,
                key_column=catalog.key_column,
                path_column=catalog.path_column,
            )
            rows.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "style": str(row[catalog.style_column]),
                    "gender": str(row[catalog.gender_column]),
                    "image_key": img_key,
                    "image_path": str(img_path) if img_path else "",
                }
            )
        return pd.DataFrame(rows)


class StreamlitRecommendationApp:
    def __init__(self):
        try:
            import streamlit as st
        except ImportError as e:
            raise RuntimeError("Streamlit is not installed. Install with: pip install streamlit") from e
        self.st = st

    def run(self) -> None:
        st = self.st
        st.set_page_config(page_title="Fashion Recommender", layout="wide")
        st.title("Fashion Recommendation")
        st.caption("Image feature + survey vector adjustment + FAISS/embedding retrieval")

        settings = self._render_sidebar()

        @st.cache_resource(show_spinner=True)
        def _cached_load(
            metadata_csv_: str,
            embeddings_npy_: str,
            faiss_index_: str,
            model_checkpoint_: str,
            label_map_json_: str,
            survey_csv_: str,
            use_survey_csv_prior_: bool,
            num_classes_: int,
            device_name_: str,
            image_size_: int,
            transform_profile_: str,
        ) -> LoadedResources:
            return ResourceFactory.build_loaded_resources(
                metadata_csv=metadata_csv_,
                embeddings_npy=embeddings_npy_,
                faiss_index=faiss_index_,
                model_checkpoint=model_checkpoint_,
                label_map_json=label_map_json_,
                survey_csv=survey_csv_,
                use_survey_csv_prior=use_survey_csv_prior_,
                num_classes=num_classes_,
                device_name=device_name_,
                image_size=image_size_,
                transform_profile=transform_profile_,
                dropout=ResourceFactory.DEFAULT_DROPOUT,
            )

        loaded: Optional[LoadedResources] = None
        if not settings["preview_mode"]:
            try:
                loaded = _cached_load(
                    settings["metadata_csv"],
                    settings["embeddings_npy"], # 燁삳똾源됪에?볥젃 ???筌왖??쇱벥 feature vector
                    settings["faiss_index"],
                    settings["model_checkpoint"],# 筌뤴뫀??揶쎛餓λ쵐??
                    settings["label_map_json"],
                    settings["survey_csv"], # ??뿅??怨쀪텢 野껉퀗??
                    bool(settings["use_survey_csv_prior"]), 
                    int(settings["num_classes"]),# ??곌볼 嚥≪뮆諭???쎈솭??揶쎛?紐꾩궞 ??됱뇚筌ｌ꼶??????
                    settings["device_name"], # ?곕뗀以?筌뤴뫀諭?
                    int(settings["image_size"]), # ???筌왖 ??由?
                    settings["transform_profile"], # ?곕뗀以?????transform
                )
            except Exception as e:
                st.error(f"Resource loading failed: {e}")
                st.info("Enable preview mode in the sidebar to inspect UI without resources.")
                st.stop()

        style_options = self._resolve_style_options(loaded)
        self._render_recommendation_ui(loaded, settings, style_options)

    # SideBar UI 
    def _render_sidebar(self) -> Dict[str, object]:
        st = self.st
        with st.sidebar:
            st.subheader("Resources")
            metadata_csv = st.text_input("카탈로그 정보", value="catalog_metadata.csv")
            embeddings_npy = st.text_input("임베딩 벡터 정보 (.npy/.npz)", value="catalog_features.npy")
            faiss_index = st.text_input("FAISS Index", value="")
            model_checkpoint = st.text_input(
                "모델 가중치",
                value="best_model_state.pth",
            )
            label_map_json = st.text_input(
                "라벨링 정보",
                value="outputs_resnet50/label_to_index.json",
            )
            survey_csv = st.text_input(
                "설문조사 데이터",
                value="survey_result_all.csv",
            )
            catalog_root_text = st.text_input("Catalog image root (optional)", value="")
            st.divider()
            num_classes = st.number_input("Fallback num_classes", min_value=2, value=31, step=1)
            image_size = st.number_input("Image size", min_value=112, max_value=512, value=224, step=16)
            transform_profile = st.selectbox("Transform profile", ["custom", "square"], index=0)
            use_survey_csv_prior = st.checkbox("Use survey CSV prior vector", value=True)
            device_name = st.selectbox("Device", ["cuda", "cpu"], index=0 if torch.cuda.is_available() else 1)
            st.divider()
            preview_mode = st.checkbox("UI preview mode (skip resource loading)", value=False)

        return {
            "metadata_csv": metadata_csv,
            "embeddings_npy": embeddings_npy,
            "faiss_index": faiss_index,
            "model_checkpoint": model_checkpoint,
            "label_map_json": label_map_json,
            "survey_csv": survey_csv, 
            "catalog_root_text": catalog_root_text,
            "num_classes": int(num_classes),
            "image_size": int(image_size),
            "transform_profile": transform_profile,
            "use_survey_csv_prior": bool(use_survey_csv_prior),
            "device_name": device_name,
            "preview_mode": bool(preview_mode),
        }

    # Meta Data Load 실패시, 디폴트값 로딩
    def _resolve_style_options(self, loaded: Optional[LoadedResources]) -> List[str]:
        if loaded is not None:
            style_series = loaded.catalog.metadata[loaded.catalog.style_column].astype(str).str.strip().str.lower()
            style_options = sorted([s for s in style_series.unique().tolist() if s and s != "unknown"])
            return style_options if style_options else ["unknown"]
        return ["casual", "classic", "minimal", "sportivecasual", "street", "vintage"]

    # User Input UI
    def _render_recommendation_ui(
        self,
        loaded: Optional[LoadedResources],
        settings: Dict[str, object],
        style_options: List[str],
    ) -> None:
        st = self.st
        st.subheader("User Input")
        uploaded = st.file_uploader("Upload a user image", type=["jpg", "jpeg", "png"])

        col1, col2, col3 = st.columns(3)
        with col1:
            gender_filter = st.selectbox("Gender filter", ["all", "male", "female"], index=0)
        with col2:
            top_k = st.slider("Top-K", min_value=1, max_value=30, value=10, step=1)
        with col3:
            use_faiss = st.checkbox("Use FAISS when possible", value=True)

        liked_styles = st.multiselect("Preferred styles", options=style_options)
        disliked_styles = st.multiselect("Disliked styles", options=style_options)

        alpha_col, beta_col, gamma_col = st.columns(3)
        with alpha_col:
            alpha_query = st.slider("alpha_query", 0.0, 2.0, 1.0, 0.05)
        with beta_col:
            beta_like = st.slider("beta_like", 0.0, 2.0, 0.8, 0.05)
        with gamma_col:
            gamma_dislike = st.slider("gamma_dislike", 0.0, 2.0, 0.8, 0.05)
        
        delta_survey_prior = st.slider("delta_survey_prior", 0.0, 2.0, 1.0, 0.05)


        """예외 처리"""
        if not st.button("Run recommendation", type="primary"):
            return
       
        if uploaded is None:
            st.warning("Please upload an image.")
            st.stop()

        """User Image 업로드 부분"""
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", width=320)

        """"Preview Mode 파트"""
        if settings["preview_mode"] or loaded is None:
            self._render_preview_results(
                top_k=int(top_k),
                gender_filter=gender_filter,
                liked_styles=liked_styles,
                style_options=style_options,
            )
            st.stop()

        if loaded.survey_profile is not None:
            st.caption(
                f"Survey prior loaded: rows={loaded.survey_profile.source_rows}, "
                f"global_styles={len(loaded.survey_profile.style_scores_global)}"
            )
        elif bool(settings["use_survey_csv_prior"]):
            st.warning("Survey prior is enabled, but no survey profile was loaded.")

        recommender = Recommender(loaded)
        catalog_root = (
            Path(str(settings["catalog_root_text"]))
            if str(settings["catalog_root_text"]).strip()
            else None
        )
        result_df = recommender.recommend(
            image=image,
            liked_styles=liked_styles,
            disliked_styles=disliked_styles,
            top_k=int(top_k),
            gender_filter=gender_filter,
            use_faiss=bool(use_faiss),
            alpha_query=float(alpha_query),
            beta_like=float(beta_like),
            gamma_dislike=float(gamma_dislike),
            delta_survey_prior=float(delta_survey_prior),
            catalog_root=catalog_root,
        )
        if result_df.empty:
            st.warning("No search result. Check metadata and gender filter.")
            st.stop()

        st.subheader("Top-K results")
        st.dataframe(result_df, use_container_width=True)
        for _, row in result_df.head(5).iterrows():
            path_text = str(row["image_path"]).strip()
            if path_text and Path(path_text).exists():
                st.image(path_text, caption=f"rank={int(row['rank'])}, score={row['score']:.4f}", width=260)


    # Render Preview Cite
    def _render_preview_results(
        self,
        top_k: int,
        gender_filter: str,
        liked_styles: Sequence[str],
        style_options: Sequence[str],
    ) -> None:
        st = self.st
        st.info("Preview mode: showing mock results without model/FAISS retrieval.")
        mock_rows = []
        for rank in range(1, top_k + 1):
            mock_rows.append(
                {
                    "rank": rank,
                    "score": float(max(0.0, 1.0 - 0.03 * (rank - 1))),
                    "style": liked_styles[0] if liked_styles else style_options[(rank - 1) % len(style_options)],
                    "gender": (
                        gender_filter
                        if gender_filter in {"male", "female"}
                        else ("male" if rank % 2 else "female")
                    ),
                    "image_key": f"preview_item_{rank:03d}.jpg",
                    "image_path": "",
                }
            )
        st.subheader("Top-K results (preview)")
        st.dataframe(pd.DataFrame(mock_rows), use_container_width=True)


def is_running_in_streamlit() -> bool:
    try:
        import streamlit as st  # type: ignore
        return st.runtime.exists()  # type: ignore[attr-defined]
    except Exception:
        return False


def main() -> None:
    if is_running_in_streamlit():
        StreamlitRecommendationApp().run()
        return
    raise RuntimeError("This app is UI-only. Please run with: streamlit run app.py")


if __name__ == "__main__":
    main()
