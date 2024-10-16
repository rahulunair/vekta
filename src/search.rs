use crate::ann::RandomProjectionIndex;
use crate::config::{Number, State};
use crate::database::VectorDatabase;
use crate::vector_entry::{Metadata, VectorEntry};
use crate::vector_ops::{compute_cosine_similarity_simd, normalize_vector};
use anyhow::Result;
use std::time::Instant;

pub struct SearchTimings {
    pub search_duration: std::time::Duration,
    pub sort_duration: std::time::Duration,
    pub total_duration: std::time::Duration,
}

pub struct SearchResult {
    pub similarity: Number,
    pub label: String,
    pub unique_id: String,
    pub metadata: Metadata,
}

pub struct SearchEngine {
    pub db: VectorDatabase,
    ann_index: Option<RandomProjectionIndex>,
}

impl SearchEngine {
    pub fn new(db: VectorDatabase, state: &State) -> Result<Self> {
        let ann_index = if state.search_method == "ann" || state.search_method == "hybrid" {
            Some(Self::initialize_ann_index(&db, state)?)
        } else {
            None
        };

        Ok(Self { db, ann_index })
    }

    fn initialize_ann_index(db: &VectorDatabase, state: &State) -> Result<RandomProjectionIndex> {
        let data_size = db.count()?;
        let mut index = RandomProjectionIndex::new(state.dimensions, data_size);

        println!("Debug: Initializing ANN index");
        let mut count = 0;
        for i in 0..data_size {
            if let Some(entry) = db.get_entry_by_index(i)? {
                // Vectors should already be normalized in the database
                index.add(entry.vector.clone(), i);
                count += 1;
            }
        }
        println!("Debug: Added {} vectors to ANN index", count);

        index.print_hash_tables();

        Ok(index)
    }

    pub fn search(
        &self,
        query_vector: &[Number],
        state: &State,
    ) -> Result<(Vec<SearchResult>, SearchTimings)> {
        let start = Instant::now();

        println!("Debug: Using search method: {}", state.search_method);

        // Normalize the query vector once
        let mut normalized_query = query_vector.to_vec();
        normalize_vector(&mut normalized_query);

        let all_similarities = match state.search_method.as_str() {
            "exact" => self.exact_search(&normalized_query)?,
            "ann" => self.ann_search(&normalized_query, state.top_k)?,
            "hybrid" => self.hybrid_search(&normalized_query, state)?,
            _ => anyhow::bail!("Unknown search method: {}", state.search_method),
        };

        let sort_start = Instant::now();
        let results = self.sort_and_limit_results(all_similarities, state.top_k);
        let sort_duration = sort_start.elapsed();

        let search_duration = start.elapsed();
        println!("Debug: Search completed. Found {} results", results.len());

        let timings = SearchTimings {
            search_duration,
            sort_duration,
            total_duration: start.elapsed(),
        };

        Ok((results, timings))
    }

    fn exact_search(&self, query_vector: &[Number]) -> Result<Vec<SearchResult>> {
        println!("Debug: Performing exact search");
        let mut results = Vec::new();
        for i in 0..self.db.count()? {
            if let Some(entry) = self.db.get_entry_by_index(i)? {
                if let Some(result) = self.compute_similarity(query_vector, &entry) {
                    results.push(result);
                }
            }
        }
        Ok(results)
    }

    fn ann_search(&self, query_vector: &[Number], top_k: usize) -> Result<Vec<SearchResult>> {
        println!("Debug: Performing ANN search");
        let mut results = Vec::new();
        if let Some(index) = &self.ann_index {
            let candidate_indices = index.search(query_vector.to_vec(), top_k);
            println!(
                "Debug: ANN search returned {} candidate indices",
                candidate_indices.len()
            );
            for &i in &candidate_indices {
                if let Some(entry) = self.db.get_entry_by_index(i)? {
                    if let Some(result) = self.compute_similarity(query_vector, &entry) {
                        results.push(result);
                    }
                }
            }
        } else {
            anyhow::bail!("ANN index not initialized");
        }
        Ok(results)
    }

    fn hybrid_search(&self, query_vector: &[Number], state: &State) -> Result<Vec<SearchResult>> {
        println!("Debug: Performing hybrid search");

        let mut results = self.ann_search(query_vector, state.top_k)?;

        if results.len() < state.top_k {
            println!("Debug: ANN search found fewer than top_k results, performing exact search");
            results.extend(self.exact_search(query_vector)?);
        }

        Ok(results)
    }

    fn compute_similarity(
        &self,
        query_vector: &[Number],
        entry: &VectorEntry,
    ) -> Option<SearchResult> {
        // Assume entry.vector is already normalized
        compute_cosine_similarity_simd(query_vector, &entry.vector).map(|similarity| {
            println!(
                "Debug: Similarity for entry {}: {}",
                entry.label, similarity
            );
            SearchResult {
                similarity,
                label: entry.label.clone(),
                unique_id: entry.unique_id.clone(),
                metadata: entry.metadata.clone(),
            }
        })
    }

    fn sort_and_limit_results(
        &self,
        mut results: Vec<SearchResult>,
        limit: usize,
    ) -> Vec<SearchResult> {
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        results
    }
}
