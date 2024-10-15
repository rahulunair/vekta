import sys
import json
import random
import subprocess
import os
import time
from loguru import logger
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style, init

# Initialize colorama for cross-platform color support
init()

# Set up logger
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

# Vekta configuration
VEKTA_PATH = "test_db.bin"
VEKTA_DIMENSIONS = "384"
VEKTA_LABEL_SIZE = "64"  # Increased to accommodate longer labels
VEKTA_CHUNK_SIZE = "2048"  # Adjust if needed based on your metadata size

def set_vekta_env():
    os.environ["VEKTA_PATH"] = VEKTA_PATH
    os.environ["VEKTA_DIMENSIONS"] = VEKTA_DIMENSIONS
    os.environ["VEKTA_LABEL_SIZE"] = VEKTA_LABEL_SIZE
    os.environ["VEKTA_CHUNK_SIZE"] = VEKTA_CHUNK_SIZE

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
    for embedding in embeddings:
        run_vekta("add", json.dumps(embedding))

def parse_search_results(output: str) -> Dict[str, Any]:
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON output: {output}")
        return {"results": [], "actual_results_count": 0}

def perform_search(embedding: Dict[str, Any], top_k: int, search_method: str) -> Dict[str, Any]:
    start_time = time.time()
    search_result = run_vekta("search", json.dumps(embedding), top_k=top_k, search_method=search_method)
    end_time = time.time()
    
    parsed_results = parse_search_results(search_result)
    
    return {
        "time": end_time - start_time,
        "result_count": parsed_results["actual_results_count"],
        "results": parsed_results["results"]
    }

def print_search_results(query: Dict[str, Any], results: List[Dict[str, Any]], method: str):
    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Query ({method}):{Style.RESET_ALL} {query['metadata']['content_preview']}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    
    for i, result in enumerate(results, 1):
        similarity = result['similarity']
        content = result['metadata']['content_preview']
        print(f"{Fore.YELLOW}Result {i}:{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}Similarity:{Style.RESET_ALL} {similarity:.4f}")
        print(f"{Fore.BLUE}Content:{Style.RESET_ALL} {content}")
        print(f"{Fore.CYAN}{'-'*80}{Style.RESET_ALL}")

def compare_search_methods(embeddings: List[Dict[str, Any]], query_embeddings: List[Dict[str, Any]], top_k: int):
    logger.info(f"Comparing search methods with TOP_K={top_k}")
    
    results = {}
    for method in ["exact", "ann"]:
        times = []
        result_counts = []
        
        for query in query_embeddings:
            result = perform_search(query, top_k, method)
            
            times.append(result["time"])
            result_counts.append(result["result_count"])
            
            logger.info(f"{method.upper()} search for {query['label']}:")
            logger.info(f"  Time: {result['time']:.4f}s, Results: {result['result_count']}")
            
            # Print similar texts for each query
            print_search_results(query, result["results"], method.upper())
        
        avg_time = sum(times) / len(times)
        avg_result_count = sum(result_counts) / len(result_counts)
        logger.info(f"{method.upper()} Average time: {avg_time:.4f}s")
        logger.info(f"{method.upper()} Average result count: {avg_result_count:.2f}")
        
        results[method] = {"times": times, "result_counts": result_counts, "avg_time": avg_time, "avg_result_count": avg_result_count}
    
    return results

def plot_performance(results, top_k):
    plt.figure(figsize=(12, 6))
    plt.boxplot([results["exact"]["times"], results["ann"]["times"]], labels=["Exact", "ANN"])
    plt.title(f"Search Performance (TOP_K={top_k})")
    plt.ylabel("Time (seconds)")
    plt.savefig(f"search_performance_top_k_{top_k}.png")
    plt.close()

def evaluate_search_quality(embeddings: List[Dict[str, Any]], query_embeddings: List[Dict[str, Any]], top_k: int):
    logger.info(f"Evaluating search quality with TOP_K={top_k}")
    
    exact_results = []
    ann_results = []
    
    for query in query_embeddings:
        exact_search = perform_search(query, top_k, "exact")
        ann_search = perform_search(query, top_k, "ann")
        
        exact_labels = set(result["label"] for result in exact_search["results"])
        ann_labels = set(result["label"] for result in ann_search["results"])
        
        overlap = len(exact_labels.intersection(ann_labels))
        exact_results.append(len(exact_labels))
        ann_results.append(overlap)
    
    accuracy = sum(ann_results) / sum(exact_results)
    logger.info(f"ANN search accuracy: {accuracy:.2%}")
    
    return accuracy

def main():
    set_vekta_env()
    
    # Load embeddings
    embeddings = load_embeddings("embeddings.jsonl")
    query_embeddings = load_embeddings("query_embeddings.jsonl")
    
    # Remove existing database file if it exists
    if os.path.exists(VEKTA_PATH):
        os.remove(VEKTA_PATH)
    
    # Add embeddings to Vekta
    logger.info("Adding embeddings to Vekta...")
    start_time = time.time()
    add_embeddings(embeddings)
    end_time = time.time()
    embedding_time = end_time - start_time
    logger.info(f"Time taken to add {len(embeddings)} embeddings: {embedding_time:.4f}s")
    
    # Compare search methods
    top_k_values = [5, 10, 20, 50, 100]
    performance_results = {}
    quality_results = {}
    
    for top_k in top_k_values:
        performance_results[top_k] = compare_search_methods(embeddings, query_embeddings[:5], top_k)  # Use only first 5 queries for brevity
        plot_performance(performance_results[top_k], top_k)
        quality_results[top_k] = evaluate_search_quality(embeddings, query_embeddings, top_k)
    
    # Print summary
    logger.info("\n--- SUMMARY ---")
    logger.info(f"Number of embeddings: {len(embeddings)}")
    logger.info(f"Time to add embeddings: {embedding_time:.4f}s")
    logger.info("\nSearch performance:")
    for top_k in top_k_values:
        logger.info(f"TOP_K={top_k}:")
        for method in ["exact", "ann"]:
            logger.info(f"  {method.upper()}: {performance_results[top_k][method]['avg_time']:.4f}s, {performance_results[top_k][method]['avg_result_count']:.2f} results")
        logger.info(f"  ANN Quality: {quality_results[top_k]:.2%}")
    
    # Clean up: remove the database file after the test
    os.remove(VEKTA_PATH)

if __name__ == "__main__":
    main()
