# Vektā Vector Database

Vektā (वेक्ता, "expresses clearly" in Sanskrit) is a simple vector database designed for efficient storage and similarity search. It's built as a command-line tool, embracing Unix philosophy for easy integration with other tools and scripts.

## Features

- Fast vector similarity search using cosine similarity
- Support for exact search and ANN search
- SIMD-accelerated computations for improved performance
- Parallel processing using Rayon
- Configurable via environment variables or config file
- Designed for easy use with Unix pipes and scripts
- JSON input/output for easy integration with other tools

## Installation

Ensure you have Rust installed, then run:

```bash
cargo install --path .
```

## Input/Output Format

Vektā uses JSON for input and output. Each vector is represented as:

```json
{"label": "example_label", "vector": [0.1, 0.2, 0.3, ...]}
```

Search results are output in JSON format, including the query details and matched vectors with their similarities.

## Usage


Vektā supports the following commands:

### Add vectors

```bash
cat vectors.jsonl | vekta add
```

### List vectors

```bash
vekta list
```

### Search for similar vectors

```bash
echo '{"label": "query", "vector": [0.1, 0.2, 0.3, ...]}' | vekta search
```

### Print configuration

```bash
vekta config
```

## Configuration

Vektā can be configured via environment variables or a `config.toml` file. Environment variables take precedence.

### Environment Variables

- `VEKTA_PATH`: Path to the database file
- `VEKTA_DIMENSIONS`: Number of dimensions for vectors (must be a multiple of 8)
- `VEKTA_LABEL_SIZE`: Maximum size of labels in bytes
- `VEKTA_TOP_K`: Number of top results to return
- `VEKTA_SEARCH_METHOD`: Search method ("exact" or "ann")
- `VEKTA_ANN_NUM_PROJECTIONS`: Number of projections for ANN search
- `VEKTA_VERBOSE`: Enable verbose output ("true" or "false")

### Config File

Create a `config.toml` file in the same directory as the executable:

```toml
path = "path/to/database.bin"
dimensions = 384
label_size = 32
top_k = 10
search_method = "ann"
ann_num_projections = 20
```


## Use Cases

Vektā is particularly useful for:

1. Semantic search applications
2. Recommendation systems
3. Image similarity search
4. Document clustering and classification

It can be used with various types of vector embeddings, such as:

- Text embeddings (e.g., from BERT, GPT, or other language models)
- Image embeddings (e.g., from CNN models)
- Audio embeddings
- User behavior embeddings

## Example: Using with Sentence Transformers

Here's an example of how to use Vektā with Sentence Transformers for semantic search:

```python
import sys
import json
import random
import subprocess
import os
import time
from loguru import logger
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# Set up logger and Vekta configuration
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

VEKTA_PATH = "test_db.bin"
VEKTA_DIMENSIONS = "384"
VEKTA_LABEL_SIZE = "32"

def set_vekta_env():
    os.environ["VEKTA_PATH"] = VEKTA_PATH
    os.environ["VEKTA_DIMENSIONS"] = VEKTA_DIMENSIONS
    os.environ["VEKTA_LABEL_SIZE"] = VEKTA_LABEL_SIZE

def run_vekta(command: str, input_data: str = None, top_k: int = None, search_method: str = None) -> str:
    env = os.environ.copy()
    if top_k is not None:
        env["VEKTA_TOP_K"] = str(top_k)
    if search_method is not None:
        env["VEKTA_SEARCH_METHOD"] = search_method
    
    process = subprocess.Popen(
        ["./target/release/vekta", command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    stdout, stderr = process.communicate(input=input_data)
    if process.returncode != 0:
        logger.error(f"Error running Vekta {command}:\n{stderr}")
        exit(1)
    return stdout

def add_embeddings(embeddings: List[Dict[str, Any]]):
    added_labels = set()
    for embedding in embeddings:
        if embedding['label'] not in added_labels:
            run_vekta("add", json.dumps(embedding))
            added_labels.add(embedding['label'])
        else:
            logger.warning(f"Duplicate label '{embedding['label']}' found. Skipping.")

def perform_search(embedding: Dict[str, Any], top_k: int, search_method: str) -> Dict[str, Any]:
    start_time = time.time()
    search_result = run_vekta("search", json.dumps(embedding), top_k=top_k, search_method=search_method)
    end_time = time.time()
    
    return {
        "time": end_time - start_time,
        "output": search_result,
    }

def main():
    set_vekta_env()
    
    # Load a pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode some sentences
    sentences = [
        "The cat sits on the mat",
        "Dogs are man's best friend",
        "It's raining cats and dogs"
    ]
    embeddings = model.encode(sentences)

    # Add vectors to Vektā
    logger.info("Adding embeddings to Vekta...")
    start_time = time.time()
    for sentence, embedding in zip(sentences, embeddings):
        vector_data = {"label": sentence, "vector": embedding.tolist()}
        add_embeddings([vector_data])
    end_time = time.time()
    embedding_time = end_time - start_time
    logger.info(f"Time taken to add {len(sentences)} embeddings: {embedding_time:.4f}s")

    # Perform a search
    query = "feline on a rug"
    query_embedding = model.encode([query])[0]
    search_data = {"label": query, "vector": query_embedding.tolist()}
    
    for method in ["exact", "ann"]:
        logger.info(f"Performing {method.upper()} search...")
        search_result = perform_search(search_data, 5, method)
        logger.info(f"Search results ({method}):")
        logger.info(search_result["output"])
        logger.info(f"Search time: {search_result['time']:.4f}s")

if __name__ == "__main__":
    main()
```

## Performance Considerations

- Vektā uses SIMD instructions for faster vector computations. Ensure your CPU supports the required SIMD features.
- The ANN search method can significantly speed up searches on large datasets, with a trade-off in accuracy.
- Adjust the `VEKTA_ANN_NUM_PROJECTIONS` to balance between search speed and accuracy when using ANN.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.