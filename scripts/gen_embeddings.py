from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from loguru import logger

def create_embeddings(dataset_name, split, text_column, num_samples=1000, output_file='embeddings.jsonl'):
    logger.info(f"Starting embedding generation process")
    logger.info(f"Dataset: {dataset_name}, Split: {split}, Text column: {text_column}")
    logger.info(f"Number of samples: {num_samples}")

    logger.info("Loading dataset")
    dataset = load_dataset(dataset_name, split=split)
    logger.success("Dataset loaded successfully")

    logger.info("Loading SentenceTransformer model")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    logger.success("Model loaded successfully")

    logger.info("Sampling and shuffling dataset")
    sampled_data = dataset.shuffle(seed=42).select(range(num_samples))
    logger.success(f"Sampled {len(sampled_data)} entries from dataset")

    logger.info("Generating embeddings")
    embeddings = model.encode(sampled_data[text_column], convert_to_numpy=True, show_progress_bar=True)
    logger.success(f"Generated {len(embeddings)} embeddings")

    logger.info(f"Saving embeddings to {output_file}")
    with open(output_file, 'w') as f:
        for i, embedding in enumerate(embeddings):
            json.dump({"label": f"text_{i}", "vector": embedding.tolist()}, f)
            f.write('\n')
    logger.success(f"Saved {len(embeddings)} embeddings to {output_file}")

    logger.info("Embedding generation process completed")

if __name__ == "__main__":
    logger.add("embedding_generation.log", rotation="10 MB")
    create_embeddings('ag_news', 'train', 'text', num_samples=1000)