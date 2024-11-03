use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use std::collections::{HashMap, HashSet};

use crate::config::Number;
use crate::vector_ops::compute_cosine_similarity_simd;
use crate::database::VectorDatabase;

const SEED: u64 = 42;
const MIN_PROJECTIONS: usize = 2;
const MAX_PROJECTIONS: usize = 16;
const MIN_TABLES: usize = 1;
const MAX_TABLES: usize = 8;

pub struct RandomProjectionIndex {
    random_vectors: Vec<Vec<Number>>,
    hash_tables: Vec<HashMap<u64, Vec<usize>>>,
    num_tables: usize,
    num_projections: usize,
    db: VectorDatabase,
}

impl RandomProjectionIndex {
    pub fn new(dim: usize, data_size: usize, db: &VectorDatabase) -> Self {
        let (num_projections, num_tables) = Self::calculate_params(data_size);
        println!(
            "Debug: Using {} projections and {} tables",
            num_projections, num_tables
        );

        let mut rng = StdRng::seed_from_u64(SEED);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let random_vectors: Vec<Vec<Number>> = (0..num_tables * num_projections)
            .map(|_| normal.sample_iter(&mut rng).take(dim).collect())
            .collect();

        RandomProjectionIndex {
            random_vectors,
            hash_tables: vec![HashMap::new(); num_tables],
            num_tables,
            num_projections,
            db: db.clone(),
        }
    }

    fn calculate_params(data_size: usize) -> (usize, usize) {
        let log_size = (data_size as f64).log2() as usize;
        let num_projections = (log_size + 2).clamp(MIN_PROJECTIONS, MAX_PROJECTIONS);
        let num_tables = (log_size / 2 + 2).clamp(MIN_TABLES, MAX_TABLES);
        (num_projections, num_tables)
    }

    fn hash_vector(&self, vector: &[Number], table_index: usize) -> u64 {
        let start = table_index * self.num_projections;
        let end = start + self.num_projections;

        if self.hash_tables[0].len() <= 4 {
            // For very small datasets
            self.random_vectors[start..end]
                .iter()
                .map(|rv| {
                    rv.iter()
                        .zip(vector.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<Number>()
                })
                .enumerate()
                .fold(0u64, |acc, (i, proj)| {
                    if proj >= -0.1 {
                        acc | (1 << i)
                    } else {
                        acc
                    } // More lenient threshold
                })
        } else {
            // Original hashing for larger datasets
            self.random_vectors[start..end]
                .iter()
                .map(|rv| {
                    rv.iter()
                        .zip(vector.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<Number>()
                })
                .enumerate()
                .fold(
                    0u64,
                    |acc, (i, proj)| {
                        if proj >= 0.0 {
                            acc | (1 << i)
                        } else {
                            acc
                        }
                    },
                )
        }
    }

    pub fn add(&mut self, vector: Vec<Number>, index: usize) {
        for i in 0..self.num_tables {
            let hash = self.hash_vector(&vector, i);
            println!(
                "Debug: Adding vector {} to table {}, hash: {}",
                index, i, hash
            );
            self.hash_tables[i]
                .entry(hash)
                .or_insert_with(Vec::new)
                .push(index);
        }
    }

    pub fn search(&self, query: Vec<Number>, k: usize) -> Vec<usize> {
        let mut candidates = HashSet::new();
        let similarity_threshold = 0.5; // Adjust this value as needed

        for i in 0..self.num_tables {
            let query_hash = self.hash_vector(&query, i);
            println!(
                "Debug: Checking hash table {}, query hash: {}",
                i, query_hash
            );

            // Check the exact hash bucket
            if let Some(bucket) = self.hash_tables[i].get(&query_hash) {
                println!("Debug: Found {} candidates in bucket", bucket.len());
                candidates.extend(bucket);
            }

            // Check neighboring buckets (multi-probe)
            for j in 0..self.num_projections {
                for flip in 0..2 {
                    let neighbor_hash = query_hash ^ (flip << j);
                    if let Some(bucket) = self.hash_tables[i].get(&neighbor_hash) {
                        println!(
                            "Debug: Found {} candidates in neighboring bucket (hash: {})",
                            bucket.len(),
                            neighbor_hash
                        );
                        candidates.extend(bucket);
                    }
                }
            }
        }

        // Filter candidates based on similarity
        let filtered_candidates: Vec<usize> = candidates
            .into_iter()
            .filter(|&index| {
                if let Ok(Some(entry)) = self.db.get_entry_by_index(index) {
                    let similarity = compute_cosine_similarity_simd(&query, &entry.vector);
                    similarity >= similarity_threshold
                } else {
                    false
                }
            })
            .collect();

        filtered_candidates.into_iter().take(k).collect()
    }

    pub fn print_hash_tables(&self) {
        for (i, table) in self.hash_tables.iter().enumerate() {
            println!("Hash Table {}:", i);
            for (hash, indices) in table {
                println!("  Hash {}: {:?}", hash, indices);
            }
        }
    }
}

