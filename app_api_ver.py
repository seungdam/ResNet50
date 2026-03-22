from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import requests
from PIL import Image


def normalize_gender(value: str) -> str:
    v = str(value).strip().lower()
    if v in {"man", "men", "m", "male"}:
        return "male"
    if v in {"woman", "women", "w", "f", "female"}:
        return "female"
    return "all"


def parse_csv_list(text: str) -> List[str]:
    raw = str(text).strip()
    if not raw:
        return []
    return [token.strip().lower() for token in raw.split(",") if token.strip()]


def join_url(base: str, path: str) -> str:
    base = str(base).rstrip("/")
    path = str(path).strip()
    if not path:
        path = "/search"
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{base}{path}"


def extract_results(payload: object) -> List[Dict[str, object]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("results", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def get_image_ref(item: Dict[str, object]) -> str:
    for key in ("image_url", "url", "thumbnail_url", "image_path", "path"):
        val = item.get(key)
        if val is None:
            continue
        txt = str(val).strip()
        if txt:
            return txt
    return ""


class StreamlitApiRecommendationApp:
    def __init__(self) -> None:
        try:
            import streamlit as st
        except ImportError as exc:
            raise RuntimeError("Streamlit is not installed. Install with: pip install streamlit") from exc
        self.st = st

    def run(self) -> None:
        st = self.st
        st.set_page_config(page_title="Fashion Recommender (API)", layout="wide")
        st.title("Fashion Recommendation (API Client)")
        st.caption("Calls backend /search endpoint and renders results")

        settings = self._render_sidebar()
        style_options = self._resolve_style_options(settings)
        self._render_main_ui(settings, style_options)

    def _render_sidebar(self) -> Dict[str, object]:
        st = self.st
        with st.sidebar:
            st.subheader("API Settings")
            api_base = st.text_input("API Base URL", value="http://127.0.0.1:8000")
            search_path = st.text_input("Search Path", value="/search")
            timeout_sec = st.number_input("Request timeout (sec)", min_value=5, max_value=600, value=60, step=5)
            verify_ssl = st.checkbox("Verify SSL", value=True)
            preview_mode = st.checkbox("Preview mode (no API call)", value=False)

            st.divider()
            st.subheader("Style Options")
            auto_fetch_styles = st.checkbox("Fetch styles from API (/styles)", value=False)
            style_csv = st.text_area(
                "Manual style list (comma separated)",
                value="casual,street,classic,minimal,sporty,formal,vintage",
            )

            st.divider()
            st.subheader("Optional Header")
            auth_token = st.text_input("Bearer token (optional)", value="", type="password")

        return {
            "api_base": str(api_base).strip(),
            "search_path": str(search_path).strip(),
            "timeout_sec": int(timeout_sec),
            "verify_ssl": bool(verify_ssl),
            "preview_mode": bool(preview_mode),
            "auto_fetch_styles": bool(auto_fetch_styles),
            "style_csv": str(style_csv),
            "auth_token": str(auth_token).strip(),
        }

    def _resolve_style_options(self, settings: Dict[str, object]) -> List[str]:
        manual = parse_csv_list(str(settings["style_csv"]))
        if settings["preview_mode"] or not settings["auto_fetch_styles"]:
            return manual if manual else ["casual", "street"]

        st = self.st
        styles_url = join_url(str(settings["api_base"]), "/styles")
        headers: Dict[str, str] = {}
        if settings["auth_token"]:
            headers["Authorization"] = f"Bearer {settings['auth_token']}"

        try:
            resp = requests.get(
                styles_url,
                timeout=int(settings["timeout_sec"]),
                verify=bool(settings["verify_ssl"]),
                headers=headers or None,
            )
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, list):
                styles = [str(x).strip().lower() for x in payload if str(x).strip()]
            elif isinstance(payload, dict):
                raw = payload.get("styles", [])
                styles = [str(x).strip().lower() for x in raw if str(x).strip()]
            else:
                styles = []

            if styles:
                return sorted(list(dict.fromkeys(styles)))
            return manual if manual else ["casual", "street"]
        except Exception as exc:
            st.warning(f"Failed to fetch /styles. Fallback to manual list. ({exc})")
            return manual if manual else ["casual", "street"]

    def _build_request(
        self,
        uploaded,
        gender_filter: str,
        top_k: int,
        preferred_styles: Sequence[str],
        disliked_styles: Sequence[str],
        fallback_fill: bool,
        settings: Dict[str, object],
    ) -> Tuple[str, Dict[str, Tuple[str, bytes, str]], Dict[str, str], Dict[str, str]]:
        url = join_url(str(settings["api_base"]), str(settings["search_path"]))

        content_type = uploaded.type or "application/octet-stream"
        files = {"image": (uploaded.name, uploaded.getvalue(), content_type)}

        data = {
            "gender": normalize_gender(gender_filter),
            "top_k": str(int(top_k)),
            "preferred_styles": ",".join([s.strip().lower() for s in preferred_styles if str(s).strip()]),
            "disliked_styles": ",".join([s.strip().lower() for s in disliked_styles if str(s).strip()]),
            "fallback_fill": "true" if fallback_fill else "false",
        }

        headers: Dict[str, str] = {}
        if settings["auth_token"]:
            headers["Authorization"] = f"Bearer {settings['auth_token']}"

        return url, files, data, headers

    def _render_main_ui(self, settings: Dict[str, object], style_options: List[str]) -> None:
        st = self.st

        st.subheader("User Input")
        uploaded = st.file_uploader("Upload user image", type=["jpg", "jpeg", "png"])

        col1, col2, col3 = st.columns(3)
        with col1:
            gender_filter = st.selectbox("Gender", ["all", "male", "female"], index=0)
        with col2:
            top_k = st.slider("Top-K", min_value=1, max_value=50, value=5, step=1)
        with col3:
            fallback_fill = st.checkbox("fallback_fill", value=True)

        preferred_styles = st.multiselect("Preferred styles", options=style_options)
        disliked_styles = st.multiselect("Disliked styles", options=style_options)

        if not st.button("Run search", type="primary"):
            return

        if uploaded is None:
            st.warning("Please upload an image.")
            return

        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", width=320)

        if settings["preview_mode"]:
            self._render_preview_results(
                top_k=int(top_k),
                gender_filter=gender_filter,
                preferred_styles=preferred_styles,
                style_options=style_options,
            )
            return

        try:
            url, files, data, headers = self._build_request(
                uploaded=uploaded,
                gender_filter=gender_filter,
                top_k=int(top_k),
                preferred_styles=preferred_styles,
                disliked_styles=disliked_styles,
                fallback_fill=bool(fallback_fill),
                settings=settings,
            )
            with st.spinner("Calling backend /search ..."):
                resp = requests.post(
                    url,
                    files=files,
                    data=data,
                    timeout=int(settings["timeout_sec"]),
                    verify=bool(settings["verify_ssl"]),
                    headers=headers or None,
                )
                resp.raise_for_status()
                payload = resp.json()
        except Exception as exc:
            st.error(f"API request failed: {exc}")
            return

        rows = extract_results(payload)
        if len(rows) == 0:
            st.warning("Response contains no results.")
            st.json(payload)
            return

        normalized_rows: List[Dict[str, object]] = []
        for idx, row in enumerate(rows, start=1):
            rank = int(row.get("rank", idx))
            score = row.get("score", row.get("similarity", ""))
            style = row.get("style", "")
            gender = row.get("gender", "")
            image_key = row.get("image_key", row.get("image_id", row.get("id", "")))
            image_ref = get_image_ref(row)
            normalized_rows.append(
                {
                    "rank": rank,
                    "score": score,
                    "style": style,
                    "gender": gender,
                    "image_key": image_key,
                    "image_ref": image_ref,
                }
            )

        result_df = pd.DataFrame(normalized_rows)
        st.subheader("Top-K results")
        st.dataframe(result_df, use_container_width=True)

        for _, row in result_df.head(5).iterrows():
            ref = str(row.get("image_ref", "")).strip()
            if not ref:
                continue
            caption = f"rank={row['rank']}, score={row['score']}"
            if ref.startswith(("http://", "https://")):
                st.image(ref, caption=caption, width=260)
            else:
                p = Path(ref)
                if p.exists():
                    st.image(str(p), caption=caption, width=260)

    def _render_preview_results(
        self,
        top_k: int,
        gender_filter: str,
        preferred_styles: Sequence[str],
        style_options: Sequence[str],
    ) -> None:
        st = self.st
        st.info("Preview mode: showing mock results without API call.")
        mock_rows: List[Dict[str, object]] = []

        for rank in range(1, top_k + 1):
            style = (
                preferred_styles[0]
                if len(preferred_styles) > 0
                else style_options[(rank - 1) % len(style_options)]
            )
            gender = gender_filter if gender_filter in {"male", "female"} else ("male" if rank % 2 else "female")
            mock_rows.append(
                {
                    "rank": rank,
                    "score": float(max(0.0, 1.0 - 0.03 * (rank - 1))),
                    "style": style,
                    "gender": gender,
                    "image_key": f"preview_{rank:03d}",
                    "image_ref": "",
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
        StreamlitApiRecommendationApp().run()
        return
    raise RuntimeError("This app is UI-only. Please run with: streamlit run app_api_ver.py")


if __name__ == "__main__":
    main()
