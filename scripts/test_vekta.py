import sys
import json
import random
import subprocess
import os
import time
from loguru import logger
from typing import List, Dict, Any

# Set up logger
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

# Vekta configuration
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

def load_embeddings(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def add_embeddings(embeddings: List[Dict[str, Any]]):
    added_labels = set()
    for embedding in embeddings:
        if embedding['label'] not in added_labels:
            run_vekta("add", json.dumps(embedding))
            added_labels.add(embedding['label'])
        else:
            logger.warning(f"Duplicate label '{embedding['label']}' found. Skipping.")

def parse_search_results(output: str) -> Dict[str, Any]:
    lines = output.strip().split('\n')
    results = []
    current_result = {}
    for line in lines:
        if line.startswith("Label:"):
            if current_result:
                results.append(current_result)
                current_result = {}
            current_result["label"] = line.split(": ", 1)[1]
        elif line.startswith("Similarity:"):
            current_result["similarity"] = float(line.split(": ", 1)[1])
    if current_result:
        results.append(current_result)
    
    return {
        "results": results,
        "result_count": len(results)
    }

def perform_search(embedding: Dict[str, Any], top_k: int, search_method: str) -> Dict[str, Any]:
    start_time = time.time()
    search_result = run_vekta("search", json.dumps(embedding), top_k=top_k, search_method=search_method)
    end_time = time.time()
    
    parsed_results = parse_search_results(search_result)
    
    return {
        "time": end_time - start_time,
        "result_count": parsed_results["result_count"],
        "output": search_result,
        "results": parsed_results["results"]
    }

def compare_search_methods(embeddings: List[Dict[str, Any]], selected_embeddings: List[Dict[str, Any]], top_k: int):
    logger.info(f"Comparing search methods with TOP_K={top_k}")
    
    results = {}
    for method in ["exact", "ann"]:
        times = []
        result_counts = []
        
        for _ in range(5):  # Perform 5 searches for each method
            search_embedding = random.choice(embeddings)
            result = perform_search(search_embedding, top_k, method)
            
            times.append(result["time"])
            result_counts.append(result["result_count"])
            
            logger.info(f"{method.upper()} search for {search_embedding['label']}:")
            logger.info(f"  Time: {result['time']:.4f}s, Results: {result['result_count']}")
        
        avg_time = sum(times) / len(times)
        avg_result_count = sum(result_counts) / len(result_counts)
        logger.info(f"{method.upper()} Average time: {avg_time:.4f}s")
        logger.info(f"{method.upper()} Average result count: {avg_result_count:.2f}")
        
        results[method] = {"avg_time": avg_time, "avg_result_count": avg_result_count}
    
    return results

def main():
    set_vekta_env()
    
    # Load and select embeddings
    embeddings = load_embeddings("embeddings.jsonl")
    selected_embeddings = random.sample(embeddings, 1000)
    
    # Remove existing database file if it exists
    if os.path.exists(VEKTA_PATH):
        os.remove(VEKTA_PATH)
    
    # Add selected embeddings to Vekta
    logger.info("Adding embeddings to Vekta...")
    start_time = time.time()
    add_embeddings(selected_embeddings)
    end_time = time.time()
    embedding_time = end_time - start_time
    logger.info(f"Time taken to add {len(selected_embeddings)} embeddings: {embedding_time:.4f}s")
    
    # List first 10 added embeddings
    logger.info("Listing first 10 added embeddings:")
    list_output = run_vekta("list")
    logger.info("\n".join(list_output.split("\n")[:11]))
    
    # Compare search methods
    results_5 = compare_search_methods(embeddings, selected_embeddings, 5)
    results_10 = compare_search_methods(embeddings, selected_embeddings, 10)
    
    # Test with a known embedding
    known_embedding = selected_embeddings[0]
    logger.info(f"Performing search with a known embedding: {known_embedding['label']}")
    for method in ["exact", "ann"]:
        search_result = perform_search(known_embedding, 5, method)
        logger.info(f"Search results ({method}):")
        logger.info(search_result["output"])
        
        if f"Successfully found the query embedding '{known_embedding['label']}'" in search_result["output"]:
            logger.success(f"Test passed: Known embedding was found successfully using {method} search")
        else:
            logger.error(f"Test failed: Known embedding was not found successfully using {method} search")
    
    # Print configuration
    logger.info("Vekta Configuration:")
    logger.info(run_vekta("print-config"))
    
    # Print summary
    logger.info("\n--- SUMMARY ---")
    logger.info(f"Number of embeddings: {len(selected_embeddings)}")
    logger.info(f"Time to add embeddings: {embedding_time:.4f}s")
    logger.info("\nSearch performance:")
    logger.info("TOP_K=5:")
    for method in ["exact", "ann"]:
        logger.info(f"  {method.upper()}: {results_5[method]['avg_time']:.4f}s, {results_5[method]['avg_result_count']:.2f} results")
    logger.info("TOP_K=10:")
    for method in ["exact", "ann"]:
        logger.info(f"  {method.upper()}: {results_10[method]['avg_time']:.4f}s, {results_10[method]['avg_result_count']:.2f} results")
    
    # Clean up: remove the database file after the test
    os.remove(VEKTA_PATH)

if __name__ == "__main__":
    main()