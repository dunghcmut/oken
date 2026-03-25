from __future__ import annotations

import hashlib
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from qdrant_client import QdrantClient
from qdrant_client.http import models
from transformers import AutoModel, AutoProcessor


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "local_image_search"
VECTOR_SIZE = 768
DEFAULT_BATCH_SIZE = 15
DEFAULT_TOP_K = 5
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Hugging Face trước đây từng dùng tên i18n cho bản base 256,
# hiện trong nhiều nơi model được thay bằng bản multilingual.
PREFERRED_MODEL_NAME = "google/siglip-base-patch16-256-i18n"
FALLBACK_MODEL_NAME = "google/siglip-base-patch16-256-multilingual"


def print_step(message: str) -> None:
    print(f"\n[STEP] {message}")


def choose_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Ban da yeu cau CUDA nhung may hien tai khong co GPU CUDA.")
        return torch.device("cuda")

    if device_arg == "cpu":
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def load_siglip_model(
    device: torch.device,
    preferred_model_name: str = PREFERRED_MODEL_NAME,
):
    model_candidates = []
    for model_name in (preferred_model_name, FALLBACK_MODEL_NAME):
        if model_name and model_name not in model_candidates:
            model_candidates.append(model_name)

    last_error: Exception | None = None
    for model_name in model_candidates:
        for local_files_only in (False, True):
            try:
                processor = AutoProcessor.from_pretrained(
                    model_name,
                    local_files_only=local_files_only,
                )
                model = AutoModel.from_pretrained(
                    model_name,
                    local_files_only=local_files_only,
                )
                model.to(device)
                model.eval()
                return processor, model, model_name
            except Exception as exc:
                last_error = exc
                source_hint = "cache local" if local_files_only else "Hugging Face / cache"
                print(
                    f"[WARN] Khong load duoc model '{model_name}' tu {source_hint}: {exc}"
                )

    raise RuntimeError(
        "Khong the tai bat ky model SigLIP nao. "
        "Hay kiem tra internet hoac ten model Hugging Face."
    ) from last_error


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return F.normalize(embeddings, p=2, dim=-1)


def extract_embedding_tensor(output: Any) -> torch.Tensor:
    # Transformers thay doi kieu return giua cac version:
    # co ban tra ve Tensor, co ban tra ve BaseModelOutputWithPooling.
    if isinstance(output, torch.Tensor):
        return output

    for attr_name in ("pooler_output", "image_embeds", "text_embeds"):
        value = getattr(output, attr_name, None)
        if isinstance(value, torch.Tensor):
            return value

    if isinstance(output, (tuple, list)) and output:
        first_item = output[0]
        if isinstance(first_item, torch.Tensor):
            return first_item

    raise TypeError(
        f"Khong the rut embedding tensor tu output kieu {type(output).__name__}."
    )


def embed_texts(
    processor,
    model,
    device: torch.device,
    texts: list[str],
) -> list[list[float]]:
    inputs = processor(
        text=texts,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = _move_to_device(inputs, device)

    with torch.inference_mode(), _autocast_context(device):
        features = model.get_text_features(**inputs)
        features = extract_embedding_tensor(features)
        features = normalize_embeddings(features)

    return features.detach().cpu().tolist()


def embed_images(
    processor,
    model,
    device: torch.device,
    image_paths: list[Path],
) -> tuple[list[Path], list[list[float]]]:
    valid_paths: list[Path] = []
    images: list[Image.Image] = []

    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))
            valid_paths.append(image_path)
        except (FileNotFoundError, OSError, UnidentifiedImageError, ValueError) as exc:
            print(f"[WARN] Bo qua anh loi '{image_path}': {exc}")

    if not valid_paths:
        return [], []

    try:
        inputs = processor(images=images, return_tensors="pt")
        inputs = _move_to_device(inputs, device)

        with torch.inference_mode(), _autocast_context(device):
            features = model.get_image_features(**inputs)
            features = extract_embedding_tensor(features)
            features = normalize_embeddings(features)

        return valid_paths, features.detach().cpu().tolist()
    finally:
        for image in images:
            image.close()


def scan_image_files(root_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    recreate: bool = False,
) -> None:
    exists = client.collection_exists(collection_name=collection_name)

    if recreate and exists:
        print_step(f"Xoa collection cu '{collection_name}' de tao lai sach")
        client.delete_collection(collection_name=collection_name)
        exists = False

    if not exists:
        print_step(f"Tao collection '{collection_name}' voi vector size = {VECTOR_SIZE}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
            on_disk_payload=True,
        )
    else:
        print_step(f"Su dung collection co san '{collection_name}'")


def make_point_id(image_path: Path) -> str:
    normalized_path = str(image_path.resolve())
    path_hash = hashlib.sha1(normalized_path.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, path_hash))


def make_payload(image_path: Path) -> dict[str, Any]:
    resolved_path = image_path.resolve()
    return {
        "path": str(resolved_path),
        "filename": image_path.name,
        "extension": image_path.suffix.lower(),
        "parent_dir": str(resolved_path.parent),
    }


def semantic_query(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    limit: int,
):
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        return response.points

    if hasattr(client, "search"):
        return client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )

    raise AttributeError("QdrantClient khong ho tro query_points() hoac search().")
