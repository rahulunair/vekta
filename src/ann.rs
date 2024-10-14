use rand::thread_rng;
use rand_distr::{Distribution, Normal};

use crate::config::Number;

pub struct RandomProjectionIndex {
    random_vectors: Vec<Vec<Number>>,
    projections: Vec<Vec<Number>>,
}

impl RandomProjectionIndex {
    pub fn new(dim: usize, num_projections: usize) -> Self {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let random_vectors: Vec<Vec<Number>> = (0..num_projections)
            .map(|_| {
                let mut v: Vec<Number> = normal.sample_iter(&mut rng).take(dim).collect();
                let magnitude: Number = v.iter().map(|&x| x * x).sum::<Number>().sqrt();
                v.iter_mut().for_each(|x| *x /= magnitude);
                v
            })
            .collect();

        RandomProjectionIndex {
            random_vectors,
            projections: Vec::new(),
        }
    }

    pub fn add(&mut self, vector: &[Number]) {
        let projection: Vec<Number> = self
            .random_vectors
            .iter()
            .map(|rv| vector.iter().zip(rv.iter()).map(|(&a, &b)| a * b).sum())
            .collect();
        self.projections.push(projection);
    }

    pub fn search(&self, query: &[Number], k: usize) -> Vec<usize> {
        let query_projection: Vec<Number> = self
            .random_vectors
            .iter()
            .map(|rv| query.iter().zip(rv.iter()).map(|(&a, &b)| a * b).sum())
            .collect();

        let mut candidates: Vec<(Number, usize)> = self
            .projections
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let dist: Number = p
                    .iter()
                    .zip(query_projection.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<Number>()
                    .sqrt();
                (dist, i)
            })
            .collect();

        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        candidates.truncate(k * 50); // Check top 50*k candidates or all if less
        candidates.into_iter().map(|(_, i)| i).collect()
    }
}