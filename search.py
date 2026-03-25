from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from siglip_utils import (
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    PREFERRED_MODEL_NAME,
    QDRANT_URL,
    choose_device,
    embed_images,
    embed_texts,
    load_siglip_model,
    print_step,
    semantic_query,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tim kiem anh local trong Qdrant bang text hoac bang anh.",
    )
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument(
        "--text",
        help="Mo ta bang text de tim anh.",
    )
    query_group.add_argument(
        "--image",
        help="Duong dan anh truy van de tim anh tuong tu.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=QDRANT_URL,
        help=f"Dia chi Qdrant. Mac dinh: {QDRANT_URL}",
    )
    parser.add_argument(
        "--collection-name",
        default=COLLECTION_NAME,
        help=f"Ten collection Qdrant. Mac dinh: {COLLECTION_NAME}",
    )
    parser.add_argument(
        "--model-name",
        default=PREFERRED_MODEL_NAME,
        help=f"Ten model Hugging Face. Mac dinh: {PREFERRED_MODEL_NAME}",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Chon thiet bi suy luan. Mac dinh: auto",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"So ket qua tra ve. Mac dinh: {DEFAULT_TOP_K}",
    )
    return parser.parse_args()


def prompt_query_mode() -> tuple[str, str]:
    print("Chon kieu tim kiem:")
    print("1. Tim bang text")
    print("2. Tim bang anh")
    mode = input("Nhap 1 hoac 2: ").strip()

    if mode == "2":
        image_path = input("Nhap duong dan anh truy van: ").strip()
        return "image", image_path

    text_query = input("Nhap mo ta can tim: ").strip()
    return "text", text_query


def build_query_vector(args: argparse.Namespace, processor, model, device) -> list[float]:
    if args.text:
        return embed_texts(
            processor=processor,
            model=model,
            device=device,
            texts=[args.text],
        )[0]

    if args.image:
        image_path = Path(args.image).expanduser()
        if not image_path.exists():
            raise FileNotFoundError(f"Khong tim thay anh truy van: {image_path}")

        valid_paths, vectors = embed_images(
            processor=processor,
            model=model,
            device=device,
            image_paths=[image_path],
        )
        if not valid_paths:
            raise ValueError("Khong doc duoc anh truy van.")
        return vectors[0]

    mode, value = prompt_query_mode()
    if mode == "image":
        args.image = value
    else:
        args.text = value

    return build_query_vector(args, processor, model, device)


def main() -> None:
    args = parse_args()

    if args.top_k <= 0:
        print("[ERROR] --top-k phai lon hon 0.")
        sys.exit(1)

    print_step(f"Ket noi Qdrant tai {args.qdrant_url}")
    client = QdrantClient(url=args.qdrant_url)

    try:
        client.get_collections()
    except Exception as exc:
        print(f"[ERROR] Khong the ket noi Qdrant: {exc}")
        sys.exit(1)

    if not client.collection_exists(collection_name=args.collection_name):
        print(
            f"[ERROR] Collection '{args.collection_name}' chua ton tai. "
            "Hay chay indexing.py truoc."
        )
        sys.exit(1)

    try:
        device = choose_device(args.device)
        print_step(f"Dung device: {device}")
        print_step("Tai processor va model SigLIP")
        processor, model, loaded_model_name = load_siglip_model(
            device=device,
            preferred_model_name=args.model_name,
        )
        print(f"Model dang dung: {loaded_model_name}")
        query_vector = build_query_vector(args, processor, model, device)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print_step(f"Tim top {args.top_k} ket qua gan nhat")
    results = semantic_query(
        client=client,
        collection_name=args.collection_name,
        query_vector=query_vector,
        limit=args.top_k,
    )

    if not results:
        print("[INFO] Khong tim thay ket qua nao.")
        sys.exit(0)

    print("\nTop ket qua:")
    for rank, item in enumerate(results, start=1):
        payload = item.payload or {}
        image_path = payload.get("path", "N/A")
        print(f"{rank}. score={item.score:.4f} | path={image_path}")


if __name__ == "__main__":
    main()
