from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from loguru import logger
import random


def create_embeddings(
    dataset_name, split, text_column, num_samples=100, output_file="embeddings.jsonl"
):
    logger.info(f"Starting embedding generation process")
    logger.info(f"Dataset: {dataset_name}, Split: {split}, Text column: {text_column}")
    logger.info(f"Number of samples: {num_samples}")

    logger.info("Loading dataset")
    dataset = load_dataset(dataset_name, split=split)
    logger.success("Dataset loaded successfully")

    logger.info("Loading SentenceTransformer model")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    logger.success("Model loaded successfully")

    logger.info("Sampling and shuffling dataset")
    sampled_data = dataset.shuffle(seed=42).select(range(num_samples))
    logger.success(f"Sampled {len(sampled_data)} entries from dataset")

    logger.info("Generating embeddings")
    embeddings = model.encode(
        sampled_data[text_column], convert_to_numpy=True, show_progress_bar=True
    )
    logger.success(f"Generated {len(embeddings)} embeddings")

    logger.info(f"Saving embeddings to {output_file}")
    with open(output_file, "w") as f:
        for i, embedding in enumerate(embeddings):
            file_name = f"document_{i:05d}.txt"
            file_path = f"/data/documents/{file_name[:2]}/{file_name}"
            content = sampled_data[text_column][i]
            content_preview = content[:100]  # First 100 characters as preview

            entry = {
                "label": f"doc_{i:05d}",
                "vector": embedding.tolist(),
                "metadata": {
                    "file_path": file_path,
                    "file_name": file_name,
                    "chunk_index": 0,
                    "start_line": 0,
                    "end_line": len(content.split("\n")),
                    "content_preview": content_preview,
                },
            }
            json.dump(entry, f)
            f.write("\n")
    logger.success(f"Saved {len(embeddings)} embeddings to {output_file}")

    # Generate query embeddings
    logger.info("Generating query embeddings")
    query_embeddings = []
    for j in range(10):
        idx = random.randint(0, len(embeddings) - 1)
        query = {
            "label": f"query_{j:03d}",
            "vector": embeddings[idx].tolist(),
            "metadata": {
                "file_path": f"/queries/query_{j:03d}.txt",
                "file_name": f"query_{j:03d}.txt",
                "chunk_index": 0,
                "start_line": 0,
                "end_line": 1,
                "content_preview": sampled_data[text_column][idx][:100],
            },
        }
        query_embeddings.append(query)

    with open("query_embeddings.jsonl", "w") as f:
        for query in query_embeddings:
            json.dump(query, f)
            f.write("\n")
    logger.success(f"Saved 100 query embeddings to query_embeddings.jsonl")

    logger.info("Embedding generation process completed")


if __name__ == "__main__":
    logger.add("embedding_generation.log", rotation="10 MB")
    create_embeddings("ag_news", "train", "text", num_samples=10000)
