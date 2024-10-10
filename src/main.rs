use std::collections::HashSet;
use std::env;
use std::fs::OpenOptions;
use std::io::{self, BufRead, Write};
use std::mem::size_of;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use config::{Config, Environment, File as ConfigFile};
use memmap2::Mmap;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json as json;
use serde_json::Value;
use wide::f32x8;

type Number = f32;

const EPSILON: f32 = 1e-6;

#[derive(Parser)]
#[command(name = "vekta")]
#[command(version = "0.1")]
#[command(about = "A simple vector database", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Add,
    List,
    Search,
    Config,
}

#[derive(Deserialize)]
struct VektaConfig {
    path: Option<String>,
    dimensions: Option<usize>,
    label_size: Option<usize>,
    top_k: Option<usize>,
    search_method: Option<String>,
    ann_num_projections: Option<usize>,
}

struct State {
    path: String,
    dimensions: usize,
    label_size: usize,
    vector_size: usize,
    chunk_size: usize,
    top_k: usize,
    search_method: String,
    ann_num_projections: usize,
}

impl State {
    fn new() -> Result<Self> {
        let mut config = Config::default();
        config.merge(ConfigFile::with_name("vekta_config").required(false))?;
        config.merge(Environment::with_prefix("VEKTA"))?;

        let vekta_config: VektaConfig = config.try_into()?;

        let path = vekta_config
            .path
            .or_else(|| env::var("VEKTA_PATH").ok())
            .context("VEKTA_PATH not set in config or environment")?;

        let dimensions = vekta_config
            .dimensions
            .or_else(|| env::var("VEKTA_DIMENSIONS").ok().and_then(|s| s.parse().ok()))
            .context("VEKTA_DIMENSIONS not set in config or environment")?;

        let label_size = vekta_config
            .label_size
            .or_else(|| env::var("VEKTA_LABEL_SIZE").ok().and_then(|s| s.parse().ok()))
            .context("VEKTA_LABEL_SIZE not set in config or environment")?;

        let top_k = vekta_config
            .top_k
            .or_else(|| env::var("VEKTA_TOP_K").ok().and_then(|s| s.parse().ok()))
            .unwrap_or(10);

        let search_method = vekta_config
            .search_method
            .or_else(|| env::var("VEKTA_SEARCH_METHOD").ok())
            .unwrap_or_else(|| "exact".to_string());

        let ann_num_projections = vekta_config
            .ann_num_projections
            .or_else(|| env::var("VEKTA_ANN_NUM_PROJECTIONS").ok().and_then(|s| s.parse().ok()))
            .unwrap_or(10);

        if dimensions % 8 != 0 {
            anyhow::bail!("VEKTA_DIMENSIONS must be a multiple of 8.");
        }

        let vector_size = dimensions * size_of::<Number>();
        let chunk_size = vector_size + label_size;

        Ok(Self {
            path,
            dimensions,
            label_size,
            vector_size,
            chunk_size,
            top_k,
            search_method,
            ann_num_projections,
        })
    }

    fn print_config(&self) {
        println!("path={}", self.path);
        println!("dimensions={}", self.dimensions);
        println!("label_size={}", self.label_size);
        println!("vector_size={}", self.vector_size);
        println!("chunk_size={}", self.chunk_size);
        println!("top_k={}", self.top_k);
        println!("search_method={}", self.search_method);
        println!("ann_num_projections={}", self.ann_num_projections);
    }
}

fn parse_input_line(line: &str, state: &State) -> Result<(String, Vec<Number>)> {
    let v: Value = serde_json::from_str(line).context("Failed to parse JSON input")?;

    let label = v["label"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing or invalid 'label' field"))?
        .to_string();

    let vector: Vec<Number> = v["vector"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Missing or invalid 'vector' field"))?
        .iter()
        .map(|x| x.as_f64().ok_or_else(|| anyhow::anyhow!("Invalid vector value")))
        .collect::<Result<Vec<f64>>>()?
        .into_iter()
        .map(|x| x as Number)
        .collect();

    if vector.len() != state.dimensions {
        anyhow::bail!(
            "Vector dimension mismatch: expected {}, got {}",
            state.dimensions,
            vector.len()
        );
    }

    Ok((label, vector))
}

fn serialize_chunk(vector: &[Number], label: &str, label_size: usize) -> Result<Vec<u8>> {
    let mut chunk = Vec::with_capacity(vector.len() * size_of::<Number>() + label_size);

    // Serialize vector
    for &num in vector {
        chunk.extend(&num.to_le_bytes());
    }

    // Serialize label with padding
    let label_bytes = label.as_bytes();
    if label_bytes.len() >= label_size {
        anyhow::bail!(
            "Label '{}' is too long (max {} bytes).",
            label,
            label_size - 1
        );
    }
    chunk.extend(label_bytes);
    chunk.resize(vector.len() * size_of::<Number>() + label_size, 0); // Padding with zeros

    Ok(chunk)
}

fn append_to_db(path: &str, chunk: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("Failed to open or create database file '{}'", path))?;
    file.write_all(chunk)
        .context("Failed to write to database file")?;
    Ok(())
}

struct Database {
    mmap: Mmap,
    record_count: usize,
    vector_size: usize,
    label_size: usize,
    chunk_size: usize,
    ann_index: Option<RandomProjectionIndex>,
}

impl Database {
    fn open(state: &State) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&state.path)
            .with_context(|| format!("Failed to open or create database file '{}'", state.path))?;

        let metadata = file.metadata()?;
        let file_size = metadata.len() as usize;
        let record_count = file_size / state.chunk_size;

        let mmap = unsafe { Mmap::map(&file)? };

        let mut ann_index = if state.search_method == "ann" {
            Some(RandomProjectionIndex::new(
                state.dimensions,
                state.ann_num_projections,
            ))
        } else {
            None
        };

        if let Some(index) = &mut ann_index {
            for i in 0..record_count {
                let vector =
                    Self::get_vector_from_mmap(&mmap, i, state.vector_size, state.chunk_size)?;
                index.add(&vector);
            }
        }

        Ok(Self {
            mmap,
            record_count,
            vector_size: state.vector_size,
            label_size: state.label_size,
            chunk_size: state.chunk_size,
            ann_index,
        })
    }

    fn get_vector_from_mmap(
        mmap: &Mmap,
        index: usize,
        vector_size: usize,
        chunk_size: usize,
    ) -> Result<Vec<Number>> {
        let start = index * chunk_size;
        let end = start + vector_size;
        if end > mmap.len() {
            anyhow::bail!("Attempted to read beyond the memory map.");
        }
        let vector_bytes = &mmap[start..end];
        let vector = vector_bytes
            .chunks_exact(size_of::<Number>())
            .map(|b| {
                Ok(Number::from_le_bytes(b.try_into().map_err(|_| {
                    anyhow::anyhow!("Failed to read number from bytes")
                })?))
            })
            .collect::<Result<Vec<Number>>>()?;
        Ok(vector)
    }

    fn get_vector(&self, index: usize) -> Result<Vec<Number>> {
        let start = index * self.chunk_size;
        let end = start + self.vector_size;
        if end > self.mmap.len() {
            anyhow::bail!("Attempted to read beyond the memory map.");
        }
        let vector_bytes = &self.mmap[start..end];
        let vector = vector_bytes
            .chunks_exact(size_of::<Number>())
            .map(|b| {
                Ok(Number::from_le_bytes(b.try_into().map_err(|_| {
                    anyhow::anyhow!("Failed to read number from bytes")
                })?))
            })
            .collect::<Result<Vec<Number>>>()?;
        Ok(vector)
    }

    fn get_label(&self, index: usize) -> Result<String> {
        let start = index * self.chunk_size + self.vector_size;
        let end = start + self.label_size;
        if end > self.mmap.len() {
            anyhow::bail!("Attempted to read beyond the memory map.");
        }
        let label_bytes = &self.mmap[start..end];
        let label_end = label_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(self.label_size);
        let label = String::from_utf8_lossy(&label_bytes[..label_end]).to_string();
        Ok(label)
    }

    fn add_to_ann(&mut self, vector: &[Number]) {
        if let Some(index) = &mut self.ann_index {
            index.add(vector);
        }
        self.record_count += 1;
    }
}

fn compute_cosine_similarity_simd(a: &[Number], b: &[Number]) -> Number {
    let mut dot_product = f32x8::splat(0.0);
    let mut mag_a = f32x8::splat(0.0);
    let mut mag_b = f32x8::splat(0.0);

    let chunks = a.chunks_exact(8).zip(b.chunks_exact(8));

    for (a_chunk, b_chunk) in chunks {
        let va = f32x8::new(a_chunk.try_into().unwrap());
        let vb = f32x8::new(b_chunk.try_into().unwrap());
        dot_product += va * vb;
        mag_a += va * va;
        mag_b += vb * vb;
    }

    let mut scalar_dot_product = dot_product.reduce_add();
    let mut scalar_mag_a = mag_a.reduce_add();
    let mut scalar_mag_b = mag_b.reduce_add();

    let remainder = a.len() % 8;
    if remainder != 0 {
        for i in a.len() - remainder..a.len() {
            scalar_dot_product += a[i] * b[i];
            scalar_mag_a += a[i] * a[i];
            scalar_mag_b += b[i] * b[i];
        }
    }

    let similarity = scalar_dot_product / (scalar_mag_a.sqrt() * scalar_mag_b.sqrt());
    similarity.max(-1.0).min(1.0) // Clamp the result to [-1, 1]
}

fn normalize_vector(vector: &mut [Number]) {
    let magnitude: Number = vector.iter().map(|&x| x * x).sum::<Number>().sqrt();
    if magnitude > EPSILON {
        for x in vector.iter_mut() {
            *x /= magnitude;
        }
    }
}

fn verbose_print(message: &str) {
    if env::var("VEKTA_VERBOSE").unwrap_or_else(|_| "false".to_string()) == "true" {
        eprintln!("{}", message);
    }
}

struct RandomProjectionIndex {
    random_vectors: Vec<Vec<Number>>,
    projections: Vec<Vec<Number>>,
}

impl RandomProjectionIndex {
    fn new(dim: usize, num_projections: usize) -> Self {
        let mut rng = rand::thread_rng();
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

    fn add(&mut self, vector: &[Number]) {
        let projection: Vec<Number> = self
            .random_vectors
            .iter()
            .map(|rv| vector.iter().zip(rv.iter()).map(|(&a, &b)| a * b).sum())
            .collect();
        self.projections.push(projection);
    }

    fn search(&self, query: &[Number], k: usize) -> Vec<usize> {
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

fn add_command(state: &State) -> Result<()> {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut line = String::new();
    let mut added_labels = HashSet::new();

    while reader.read_line(&mut line)? > 0 {
        let (label, mut vector) = parse_input_line(&line, state)?;
        if added_labels.contains(&label) {
            eprintln!("Warning: Duplicate label '{}' found. Skipping.", label);
            line.clear();
            continue;
        }
        normalize_vector(&mut vector);
        let chunk = serialize_chunk(&vector, &label, state.label_size)?;
        append_to_db(&state.path, &chunk)?;
        added_labels.insert(label.clone());
        verbose_print(&format!("Added vector with label '{}'", label));
        line.clear();
    }

    // Reopen the database to update the memory map and ANN index
    let mut db = Database::open(state)?;
    for i in 0..db.record_count {
        let vector = db.get_vector(i)?;
        db.add_to_ann(&vector);
    }

    Ok(())
}

fn list_command(state: &State) -> Result<()> {
    let db = Database::open(state)?;
    for i in 0..db.record_count {
        let label = db.get_label(i)?;
        println!("{}", label);
    }
    Ok(())
}

fn search_command(state: &State) -> Result<()> {
    verbose_print("Starting search");
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let (query_label, mut query_vector) = parse_input_line(&input, state)?;

    normalize_vector(&mut query_vector);

    let db = Database::open(state)?;

    let results: Vec<(Number, String, Vec<Number>)> =
        if state.search_method == "ann" && db.ann_index.is_some() {
            verbose_print("Using ANN search method");
            if let Some(ann_index) = &db.ann_index {
                let candidate_indices = ann_index.search(&query_vector, state.top_k * 50);
                candidate_indices
                    .into_par_iter()
                    .map(|i| {
                        let mut vector = db.get_vector(i).expect("Failed to get vector");
                        let label = db.get_label(i).expect("Failed to get label");
                        normalize_vector(&mut vector);
                        let similarity = compute_cosine_similarity_simd(&query_vector, &vector);
                        (similarity, label, vector)
                    })
                    .collect()
            } else {
                eprintln!("Warning: ANN index not initialized. Falling back to exact search.");
                (0..db.record_count)
                    .into_par_iter()
                    .map(|i| {
                        let mut vector = db.get_vector(i).expect("Failed to get vector");
                        let label = db.get_label(i).expect("Failed to get label");
                        normalize_vector(&mut vector);
                        let similarity = compute_cosine_similarity_simd(&query_vector, &vector);
                        (similarity, label, vector)
                    })
                    .collect()
            }
        } else {
            verbose_print("Using exact search method");
            (0..db.record_count)
                .into_par_iter()
                .map(|i| {
                    let mut vector = db.get_vector(i).expect("Failed to get vector");
                    let label = db.get_label(i).expect("Failed to get label");
                    normalize_vector(&mut vector);
                    let similarity = compute_cosine_similarity_simd(&query_vector, &vector);
                    (similarity, label, vector)
                })
                .collect()
        };

    verbose_print(&format!(
        "All similarities: {:?}",
        results.iter().map(|(s, l, _)| (l, s)).collect::<Vec<_>>()
    ));

    let mut results = results;
    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let actual_top_k = state.top_k.min(db.record_count);
    let mut unique_results = Vec::new();
    let mut seen_labels = HashSet::new();

    for (similarity, label, _) in results
        .iter()
        .filter(|&(similarity, _, _)| *similarity >= 0.0)
    {
        if seen_labels.insert(label) {
            unique_results.push((similarity, label));
            if unique_results.len() == actual_top_k {
                break;
            }
        }
    }

    let output = json::json!({
        "query_label": query_label,
        "query_vector": &query_vector[..5.min(query_vector.len())],
        "database_record_count": db.record_count,
        "results": unique_results.iter().map(|(similarity, label)| {
            json::json!({
                "label": label,
                "similarity": similarity
            })
        }).collect::<Vec<_>>(),
        "actual_results_count": unique_results.len(),
        "requested_results_count": actual_top_k
    });

    println!("{}", json::to_string(&output)?);

    Ok(())
}

fn config_command(state: &State) -> Result<()> {
    state.print_config();
    Ok(())
}

fn main() -> Result<()> {
    let args = Cli::parse();
    let state = State::new()?;

    match args.command {
        Commands::Add => add_command(&state)?,
        Commands::List => list_command(&state)?,
        Commands::Search => search_command(&state)?,
        Commands::Config => config_command(&state)?,
    }
    Ok(())
}