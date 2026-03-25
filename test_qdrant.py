from __future__ import annotations

import sys
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "test_pro"
QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SAMPLE_TEXTS = [
    "Hệ thống IoT giám sát mực nước hồ chứa theo thời gian thực bằng cảm biến siêu âm.",
    "Nền tảng cảnh báo rò rỉ đường ống gửi thông báo về điện thoại khi áp suất giảm bất thường.",
    "Dashboard quản lý chất lượng nước hiển thị pH, độ đục và nhiệt độ theo từng khu vực.",
    "Giải pháp thu thập dữ liệu từ trạm bơm, đồng bộ lên cloud để phân tích và dự báo sự cố.",
    "Hệ thống bảo trì thông minh giúp đơn vị cấp nước phát hiện sớm nguy cơ hỏng van và máy bơm.",
]


def print_step(message: str) -> None:
    print(f"\n[STEP] {message}")


def embed_texts(model: SentenceTransformer, texts: Iterable[str]) -> list[list[float]]:
    return model.encode(list(texts), normalize_embeddings=True).tolist()


def semantic_search(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    limit: int = 3,
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


def main() -> None:
    print_step(f"Ket noi toi Qdrant tai {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL)

    try:
        client.get_collections()
    except Exception as exc:
        print(
            "Khong the ket noi toi Qdrant. Hay chac chan docker-compose da duoc chay truoc.\n"
            f"Chi tiet loi: {exc}"
        )
        sys.exit(1)

    print_step(f"Tai embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    sample_vectors = embed_texts(model, SAMPLE_TEXTS)
    vector_size = len(sample_vectors[0])

    if client.collection_exists(collection_name=COLLECTION_NAME):
        print_step(f"Collection '{COLLECTION_NAME}' da ton tai, xoa de tao moi")
        client.delete_collection(collection_name=COLLECTION_NAME)

    print_step(f"Tao collection '{COLLECTION_NAME}' voi vector size = {vector_size}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )

    print_step("Nap 5 doan van mau vao Qdrant")
    points = [
        models.PointStruct(
            id=idx + 1,
            vector=vector,
            payload={
                "topic": "he_thong_iot_giam_sat_nuoc",
                "text": text,
            },
        )
        for idx, (text, vector) in enumerate(zip(SAMPLE_TEXTS, sample_vectors))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)

    print("\nDu lieu da san sang.")
    query = input("Nhap cau truy van semantic search: ").strip()
    if not query:
        print("Ban chua nhap truy van, dung chuong trinh.")
        sys.exit(0)

    print_step("Embed cau truy van va tim kiem top 3 ket qua gan nhat")
    query_vector = embed_texts(model, [query])[0]
    results = semantic_search(
        client=client,
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=3,
    )

    print("\nTop 3 ket qua:")
    for rank, item in enumerate(results, start=1):
        text = item.payload.get("text", "") if item.payload else ""
        print(f"{rank}. score={item.score:.4f} | text={text}")


if __name__ == "__main__":
    main()
