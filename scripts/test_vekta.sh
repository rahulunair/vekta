#!/bin/bash

set -e

# Set the path to the vekta binary
VEKTA_BIN="./target/release/vekta"

# Function to generate random vectors
generate_random_vector() {
    local dim=$1
    local vector=""
    for ((i=0; i<dim; i++)); do
        vector+="$(printf "%.6f" $(bc -l <<< "scale=6; $RANDOM/32768"))"
        if [ $i -lt $((dim-1)) ]; then
            vector+=","
        fi
    done
    echo "[$vector]"
}

# Configuration
export VEKTA_PATH="./test_db.bin"
export VEKTA_DIMENSIONS=256
export VEKTA_LABEL_SIZE=64
export VEKTA_TOP_K=2
export VEKTA_SEARCH_METHOD="exact"
export VEKTA_ANN_NUM_PROJECTIONS=10
export VEKTA_VERBOSE="true"

echo "Testing Vekta Vector Database"

# Test config command
echo "Testing config command..."
$VEKTA_BIN config

# Test add command
echo "Testing add command..."
for i in {1..6}; do
    label="test_vector_$i"
    vector=$(generate_random_vector $VEKTA_DIMENSIONS)
    echo "{\"label\":\"$label\",\"vector\":$vector,\"metadata\":{\"file_path\":\"/path/to/file_$i.txt\",\"file_name\":\"file_$i.txt\",\"chunk_index\":0,\"start_line\":0,\"end_line\":10,\"content_preview\":\"This is a test vector\"}}" | $VEKTA_BIN add
done

# Save one of the added vectors for later use
saved_vector=$(generate_random_vector $VEKTA_DIMENSIONS)
echo "{\"label\":\"saved_vector\",\"vector\":$saved_vector,\"metadata\":{\"file_path\":\"/path/to/saved.txt\",\"file_name\":\"saved.txt\",\"chunk_index\":0,\"start_line\":0,\"end_line\":10,\"content_preview\":\"This is a saved vector\"}}" | $VEKTA_BIN add

# Test list command
echo "Testing list command..."
$VEKTA_BIN list

# Test search command with exact method using a vector in the database
echo "Testing search command with exact method (existing vector)..."
echo "{\"label\":\"query\",\"vector\":$saved_vector,\"metadata\":{\"file_path\":\"/path/to/query.txt\",\"file_name\":\"query.txt\",\"chunk_index\":0,\"start_line\":0,\"end_line\":10,\"content_preview\":\"This is a query vector\"}}" | $VEKTA_BIN search

# Test search command with exact method using a random vector
echo "Testing search command with exact method (random vector)..."
random_query=$(generate_random_vector $VEKTA_DIMENSIONS)
echo "{\"label\":\"random_query\",\"vector\":$random_query,\"metadata\":{\"file_path\":\"/path/to/random_query.txt\",\"file_name\":\"random_query.txt\",\"chunk_index\":0,\"start_line\":0,\"end_line\":10,\"content_preview\":\"This is a random query vector\"}}" | $VEKTA_BIN search

# Test search command with ANN method using a vector in the database
echo "Testing search command with ANN method (existing vector)..."
export VEKTA_SEARCH_METHOD="ann"
echo "{\"label\":\"query\",\"vector\":$saved_vector,\"metadata\":{\"file_path\":\"/path/to/query.txt\",\"file_name\":\"query.txt\",\"chunk_index\":0,\"start_line\":0,\"end_line\":10,\"content_preview\":\"This is a query vector\"}}" | $VEKTA_BIN search

# Test search command with ANN method using a random vector
echo "Testing search command with ANN method (random vector)..."
echo "{\"label\":\"random_query\",\"vector\":$random_query,\"metadata\":{\"file_path\":\"/path/to/random_query.txt\",\"file_name\":\"random_query.txt\",\"chunk_index\":0,\"start_line\":0,\"end_line\":10,\"content_preview\":\"This is a random query vector\"}}" | $VEKTA_BIN search

# Clean up
rm -f $VEKTA_PATH

echo "All tests completed."