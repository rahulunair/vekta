import sys
import json
import random
import subprocess
import os
import time
import traceback
from loguru import logger
import argparse
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set up logger
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

# Vekta configuration
VEKTA_PATH = "test_db"
VEKTA_DIMENSIONS = "384"
VEKTA_LABEL_SIZE = "64"
VEKTA_CHUNK_SIZE = "2048"

# Test configuration
NUM_EMBEDDINGS = 100  # Number of embeddings to add to the database
NUM_QUERIES = 10  # Number of search queries to perform for each method

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
    
    try:
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
            logger.error(f"Error running Vekta {command}:")
            logger.error(f"Return code: {process.returncode}")
            logger.error(f"STDOUT: {stdout}")
            logger.error(f"STDERR: {stderr}")
            raise subprocess.CalledProcessError(process.returncode, process.args, stdout, stderr)
        return stdout
    except Exception as e:
        logger.error(f"Exception occurred while running Vekta {command}:")
        logger.error(traceback.format_exc())
        raise

def load_embeddings(file_path: str, num_embeddings: int = 3):
    try:
        with open(file_path, "r") as f:
            embeddings = [json.loads(line) for line in f]
            return embeddings[:num_embeddings]
    except Exception as e:
        logger.error(f"Error loading embeddings from {file_path}:")
        logger.error(traceback.format_exc())
        raise

def add_embeddings(embeddings):
    for i, embedding in enumerate(embeddings):
        try:
            json_str = json.dumps(embedding)
            run_vekta("add", json_str)
            logger.info(f"Added embedding {i + 1}")
        except Exception as e:
            logger.error(f"Error adding embedding {i + 1}:")
            logger.error(traceback.format_exc())

def parse_search_results(output: str):
    try:
        json_start = output.find('{')
        if json_start == -1:
            raise ValueError("No JSON object found in output")
        json_str = output[json_start:]
        return json.loads(json_str)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON output:")
        logger.error(output)
        return {"results": [], "actual_results_count": 0, "timings": {"search_duration_ms": 0, "sort_duration_ms": 0}}

def perform_search(embedding, top_k: int, search_method: str):
    try:
        start_time = time.time()
        search_result = run_vekta("search", json.dumps(embedding), top_k=top_k, search_method=search_method)
        end_time = time.time()
        
        parsed_results = parse_search_results(search_result)
        
        return {
            "total_time": end_time - start_time,
            "search_time": parsed_results["timings"]["search_duration_ms"] / 1000,
            "sort_time": parsed_results["timings"]["sort_duration_ms"] / 1000,
            "result_count": parsed_results["actual_results_count"],
            "results": parsed_results["results"]
        }
    except Exception as e:
        logger.error(f"Error performing search with method {search_method}:")
        logger.error(traceback.format_exc())
        return {"total_time": 0, "search_time": 0, "sort_time": 0, "result_count": 0, "results": []}

def cleanup_database():
    if os.path.exists(VEKTA_PATH):
        try:
            for file in os.listdir(VEKTA_PATH):
                file_path = os.path.join(VEKTA_PATH, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            os.rmdir(VEKTA_PATH)
            logger.info(f"Cleaned up database directory: {VEKTA_PATH}")
        except Exception as e:
            logger.error(f"Error cleaning up database directory: {VEKTA_PATH}")
            logger.error(traceback.format_exc())

def calculate_overlap(results1, results2):
    labels1 = set(result["label"] for result in results1["results"])
    labels2 = set(result["label"] for result in results2["results"])
    return len(labels1.intersection(labels2))

def plot_performance(results_df):
    plt.figure(figsize=(15, 10))
    sns.boxplot(x='Method', y='Time', hue='TimeType', data=results_df)
    plt.title('Execution Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.savefig('execution_time_comparison.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Method', y='ResultCount', data=results_df)
    plt.title('Result Count Comparison')
    plt.ylabel('Number of Results')
    plt.savefig('result_count_comparison.png')
    plt.close()

def plot_overlap(overlap_df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(overlap_df, annot=True, fmt='.2f', cmap='YlGnBu')
    plt.title('Result Overlap Heatmap (Average)')
    plt.savefig('result_overlap_heatmap.png')
    plt.close()

def main():
    set_vekta_env()
    
    # Clean up existing database
    cleanup_database()
    
    try:
        # Load embeddings
        embeddings = load_embeddings("embeddings.jsonl", NUM_EMBEDDINGS)
        
        # Add embeddings to Vekta
        logger.info(f"Adding {NUM_EMBEDDINGS} embeddings to Vekta...")
        add_embeddings(embeddings[:-NUM_QUERIES])  # Keep last NUM_QUERIES for testing
        
        # Perform searches with all three methods
        search_methods = ["exact", "ann", "hybrid"]
        all_results = []
        overlap_results = {(m1, m2): [] for m1 in search_methods for m2 in search_methods if m1 < m2}
        
        for i, query_embedding in enumerate(embeddings[-NUM_QUERIES:]):
            logger.info(f"Performing search {i+1}/{NUM_QUERIES}")
            method_results = {}
            
            for method in search_methods:
                result = perform_search(query_embedding, top_k=10, search_method=method)
                method_results[method] = result
                all_results.append({
                    'Method': method,
                    'TimeType': 'Total',
                    'Time': result['total_time'],
                    'ResultCount': result['result_count']
                })
                all_results.append({
                    'Method': method,
                    'TimeType': 'Search',
                    'Time': result['search_time'],
                    'ResultCount': result['result_count']
                })
                all_results.append({
                    'Method': method,
                    'TimeType': 'Sort',
                    'Time': result['sort_time'],
                    'ResultCount': result['result_count']
                })
            
            # Calculate overlap
            for (m1, m2) in overlap_results.keys():
                overlap = calculate_overlap(method_results[m1], method_results[m2])
                overlap_results[(m1, m2)].append(overlap)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Calculate mean overlap
        overlap_df = pd.DataFrame({f"{m1} vs {m2}": np.mean(overlaps) 
                                   for (m1, m2), overlaps in overlap_results.items()},
                                  index=search_methods)
        
        # Print performance summary
        logger.info("\nPerformance Summary:")
        summary = results_df.groupby(['Method', 'TimeType'])['Time'].agg(['mean', 'std']).reset_index()
        logger.info(tabulate(summary, headers='keys', tablefmt="grid"))
        
        # Print overlap summary
        logger.info("\nAverage Result Overlap:")
        logger.info(tabulate(overlap_df, headers='keys', tablefmt="grid"))
        
        # Generate and save plots
        plot_performance(results_df)
        plot_overlap(overlap_df)
        
        logger.info("\nPerformance plots have been saved as PNG files in the current directory.")
        
    except Exception as e:
        logger.error("An error occurred during the test:")
        logger.error(traceback.format_exc())
    finally:
        # Clean up the database after the test
        cleanup_database()

if __name__ == "__main__":
    main()
