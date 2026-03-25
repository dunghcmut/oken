from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

from siglip_utils import (
    COLLECTION_NAME,
    DEFAULT_BATCH_SIZE,
    PREFERRED_MODEL_NAME,
    QDRANT_URL,
    choose_device,
    embed_images,
    ensure_collection,
    load_siglip_model,
    make_payload,
    make_point_id,
    print_step,
    scan_image_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quet anh local va index vao Qdrant bang model SigLIP.",
    )
    parser.add_argument(
        "--image-dir",
        help="Duong dan thu muc anh can quet. Vi du: D:/Photos hoac /data/photos",
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
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"So anh xu ly moi batch. Mac dinh: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Chon thiet bi suy luan. Mac dinh: auto",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Xoa collection cu va tao lai tu dau truoc khi index.",
    )
    return parser.parse_args()


def resolve_image_dir(image_dir_arg: str | None) -> Path:
    image_dir = image_dir_arg or input("Nhap duong dan thu muc anh can index: ").strip()
    if not image_dir:
        raise ValueError("Ban chua nhap duong dan thu muc anh.")

    root_dir = Path(image_dir).expanduser()
    if not root_dir.exists():
        raise FileNotFoundError(f"Thu muc khong ton tai: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Day khong phai thu muc: {root_dir}")

    return root_dir


def main() -> None:
    args = parse_args()

    try:
        root_dir = resolve_image_dir(args.image_dir)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    if args.batch_size <= 0:
        print("[ERROR] --batch-size phai lon hon 0.")
        sys.exit(1)

    print_step(f"Ket noi Qdrant tai {args.qdrant_url}")
    client = QdrantClient(url=args.qdrant_url)

    try:
        client.get_collections()
    except Exception as exc:
        print(f"[ERROR] Khong the ket noi Qdrant: {exc}")
        sys.exit(1)

    try:
        device = choose_device(args.device)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print_step(f"Dung device: {device}")
    print_step("Tai processor va model SigLIP")
    processor, model, loaded_model_name = load_siglip_model(
        device=device,
        preferred_model_name=args.model_name,
    )
    print(f"Model dang dung: {loaded_model_name}")

    ensure_collection(
        client=client,
        collection_name=args.collection_name,
        recreate=args.recreate,
    )

    print_step(f"Quet anh trong thu muc: {root_dir}")
    image_paths = scan_image_files(root_dir)
    if not image_paths:
        print("[INFO] Khong tim thay file anh phu hop (.jpg, .jpeg, .png, .webp).")
        sys.exit(0)

    print(f"So anh tim thay: {len(image_paths)}")
    inserted_count = 0
    skipped_batches = 0

    # Index theo batch de tranh tran RAM/VRAM khi gap thu vien anh lon.
    for start_idx in tqdm(
        range(0, len(image_paths), args.batch_size),
        desc="Dang index anh",
        unit="batch",
    ):
        batch_paths = image_paths[start_idx : start_idx + args.batch_size]
        valid_paths, vectors = embed_images(
            processor=processor,
            model=model,
            device=device,
            image_paths=batch_paths,
        )

        if not valid_paths:
            skipped_batches += 1
            continue

        points = [
            models.PointStruct(
                id=make_point_id(image_path),
                vector=vector,
                payload=make_payload(image_path),
            )
            for image_path, vector in zip(valid_paths, vectors)
        ]

        client.upsert(
            collection_name=args.collection_name,
            points=points,
            wait=True,
        )
        inserted_count += len(points)

    print_step("Hoan tat index")
    print(f"So anh da upsert vao Qdrant: {inserted_count}")
    print(f"So batch bi bo qua do anh loi: {skipped_batches}")
    print(f"Collection: {args.collection_name}")
    print("Ban co the chay search.py de tim anh ngay bay gio.")


if __name__ == "__main__":
    main()
