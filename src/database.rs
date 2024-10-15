use anyhow::{Context, Result};
use memmap2::Mmap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::mem::size_of;
use std::sync::Arc;
use bincode;
use std::time::Instant;
use std::path::Path;
use crate::ann::RandomProjectionIndex;
use crate::config::{Number, State};
use crate::vector_ops::{compute_cosine_similarity_simd, normalize_vector};
use heed::EnvOpenOptions;
use heed::types::*;
use std::fs;
use std::path::PathBuf;

#[derive(Deserialize, Serialize, Clone)]
pub struct VectorEntry {
    pub label: String,
    pub vector: Vec<Number>,
    pub metadata: Metadata,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct Metadata {
    pub file_path: String,
    pub file_name: String,
    pub chunk_index: usize,
    pub start_line: usize,
    pub end_line: usize,
    pub content_preview: String,
}

pub struct LmdbWrapper {
    env: heed::Env,
    db: heed::Database<Str, SerdeBincode<Vec<u8>>>,
}

impl LmdbWrapper {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = PathBuf::from(path.as_ref());
        
        // Explicitly create the directory
        fs::create_dir_all(&path)
            .with_context(|| format!("Failed to create directory for LMDB at '{}'", path.display()))?;

        println!("Attempting to open LMDB at: {}", path.display());

        // Open the environment
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(10 * 1024 * 1024 * 1024) // 10GB
                .max_dbs(1)
                .open(&path)
                .with_context(|| format!("Failed to open LMDB environment at '{}'", path.display()))?
        };

        // Create the database
        let mut wtxn = env.write_txn()
            .with_context(|| "Failed to create write transaction for LMDB")?;
        let db: heed::Database<Str, SerdeBincode<Vec<u8>>> = env.create_database(&mut wtxn, None)
            .with_context(|| "Failed to create LMDB database")?;
        wtxn.commit()
            .with_context(|| "Failed to commit initial LMDB transaction")?;

        Ok(Self { env, db })
    }

    pub fn add(&self, key: &str, value: &[u8]) -> Result<()> {
        let mut wtxn = self.env.write_txn()?;
        self.db.put(&mut wtxn, key, &value.to_vec())?;
        wtxn.commit()?;
        Ok(())
    }

    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let rtxn = self.env.read_txn()?;
        Ok(self.db.get(&rtxn, key)?.map(|v| v.to_vec()))
    }

    pub fn iter(&self) -> Result<Vec<(String, Vec<u8>)>> {
        let rtxn = self.env.read_txn()?;
        let iter_result: Vec<(String, Vec<u8>)> = self.db
            .iter(&rtxn)?
            .map(|result| {
                result.map(|(k, v)| (k.to_string(), v.to_vec()))
            })
            .collect::<std::result::Result<Vec<_>, heed::Error>>()?;
        
        Ok(iter_result) // Return the result after the transaction is done
    }
    
}

pub struct VectorDatabase {
    mmap: Arc<Mmap>,
    pub record_count: usize,
    vector_size: usize,
    label_size: usize,
    ann_index: Option<RandomProjectionIndex>,
    offsets: Vec<usize>,
    lmdb: LmdbWrapper,
}

impl VectorDatabase {
    pub fn open(state: &State) -> Result<Self> {
        println!("Opening database at path: {}", state.path);
        // Ensure the directory exists
        let path = PathBuf::from(&state.path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("Failed to create directory for '{}'", state.path))?;
        }

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&state.path)
            .with_context(|| format!("Failed to open or create database file '{}'", state.path))?;

        let mmap = unsafe { Mmap::map(&file)? };
        let mmap = Arc::new(mmap);

        let offsets = Self::build_offsets(&mmap, state.vector_size, state.label_size)?;
        let record_count = offsets.len();

        let ann_index = if state.search_method == "ann" {
            let mut index = RandomProjectionIndex::new(state.dimensions, state.ann_num_projections);
            Self::build_ann_index(&mmap, &offsets, state.vector_size, &mut index)?;
            Some(index)
        } else {
            None
        };

        println!("Creating LMDB at path: {}_lmdb", state.path);
        let lmdb_path = format!("{}_lmdb", state.path);
        let lmdb = LmdbWrapper::new(&lmdb_path)
            .with_context(|| format!("Failed to create or open LMDB at '{}'", lmdb_path))?;

        println!("Successfully opened database and created LMDB");
        Ok(Self {
            mmap,
            record_count,
            vector_size: state.vector_size,
            label_size: state.label_size,
            ann_index,
            offsets,
            lmdb,
        })
    }

    fn build_offsets(mmap: &Mmap, vector_size: usize, label_size: usize) -> Result<Vec<usize>> {
        let mut offsets = Vec::new();
        let mut pos = 0;
        let mmap_len = mmap.len();

        while pos < mmap_len {
            offsets.push(pos);
            let metadata_length_bytes = &mmap[pos + vector_size + label_size..pos + vector_size + label_size + 4];
            let metadata_length = u32::from_le_bytes(metadata_length_bytes.try_into().unwrap()) as usize;
            let total_size = vector_size + label_size + 4 + metadata_length;
            pos += total_size;
        }

        Ok(offsets)
    }

    fn build_ann_index(mmap: &Mmap, offsets: &[usize], vector_size: usize, index: &mut RandomProjectionIndex) -> Result<()> {
        for &offset in offsets {
            let vector = Self::get_vector_from_mmap(mmap, offset, vector_size)?;
            index.add(&vector);
        }
        Ok(())
    }

    fn get_vector_from_mmap(mmap: &Mmap, offset: usize, vector_size: usize) -> Result<Vec<Number>> {
        let end = offset + vector_size;
        if end > mmap.len() {
            anyhow::bail!("Attempted to read beyond the memory map.");
        }
        let vector_bytes = &mmap[offset..end];
        let vector = vector_bytes
            .chunks_exact(size_of::<Number>())
            .map(|b| Number::from_le_bytes(b.try_into().unwrap()))
            .collect();
        Ok(vector)
    }

    pub fn get_vector(&self, index: usize) -> Result<Vec<Number>> {
        Self::get_vector_from_mmap(&self.mmap, self.offsets[index], self.vector_size)
    }

    pub fn get_label(&self, index: usize) -> Result<String> {
        let start = self.offsets[index] + self.vector_size;
        let end = start + self.label_size;
        if end > self.mmap.len() {
            anyhow::bail!("Attempted to read beyond the memory map.");
        }
        let label_bytes = &self.mmap[start..end];
        let label_end = label_bytes.iter().position(|&b| b == 0).unwrap_or(self.label_size);
        let label = String::from_utf8_lossy(&label_bytes[..label_end]).to_string();
        Ok(label)
    }

    pub fn add_to_ann(&mut self, vector: &[Number]) {
        if let Some(index) = &mut self.ann_index {
            index.add(vector);
        }
        self.record_count += 1;
    }

    pub fn get_metadata(&self, index: usize) -> Result<Metadata> {
        let start = self.offsets[index] + self.vector_size + self.label_size;
        let mut cursor = &self.mmap[start..];

        let mut length_bytes = [0u8; 4];
        length_bytes.copy_from_slice(&cursor[..4]);
        let metadata_length = u32::from_le_bytes(length_bytes) as usize;
        cursor = &cursor[4..];

        let metadata_json = std::str::from_utf8(&cursor[..metadata_length])?;
        let metadata: Metadata = serde_json::from_str(metadata_json)?;
        Ok(metadata)
    }

    pub fn add_entry_lmdb(&self, entry: &VectorEntry) -> Result<()> {
        let value = bincode::serialize(entry)?;
        self.lmdb.add(&entry.label, &value)
    }

    pub fn get_entry_lmdb(&self, label: &str) -> Result<Option<VectorEntry>> {
        if let Some(value) = self.lmdb.get(label)? {
            Ok(Some(bincode::deserialize(&value)?))
        } else {
            Ok(None)
        }
    }

    pub fn search(&self, query_vector: &[Number], state: &State) -> Result<(Vec<(Number, String, usize, Vec<Number>, Metadata)>, SearchTimings)> {
        let start = Instant::now();
        let mut timings = SearchTimings::default();

        // Memory-mapped search
        let mmap_start = Instant::now();
        let mut results = self.search_mmap(query_vector, state)?;
        timings.mmap_duration = mmap_start.elapsed();

        // LMDB search
        let lmdb_start = Instant::now();
        results.extend(self.search_lmdb(query_vector, state)?);
        timings.lmdb_duration = lmdb_start.elapsed();

        // Sort and truncate results
        let sort_start = Instant::now();
        results.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(state.top_k);
        timings.sort_duration = sort_start.elapsed();

        timings.total_duration = start.elapsed();

        Ok((results, timings))
    }

    fn search_mmap(&self, query_vector: &[Number], state: &State) -> Result<Vec<(Number, String, usize, Vec<Number>, Metadata)>> {
        let search_range = if let Some(ann_index) = self.ann_index.as_ref() {
            crate::config::verbose_print("Using ANN search method");
            ann_index.search(query_vector, state.top_k * 50)
        } else {
            crate::config::verbose_print("Using exact search method");
            (0..self.record_count).collect()
        };

        let start = Instant::now();
        let results = search_range
            .into_par_iter()
            .map(|i| -> Result<_> {
                let mut vector = self.get_vector(i)?;
                let label = self.get_label(i)?;
                let metadata = self.get_metadata(i)?;
                normalize_vector(&mut vector);
                let similarity = compute_cosine_similarity_simd(query_vector, &vector);
                Ok((similarity, label, i, vector, metadata))
            })
            .collect::<Result<Vec<_>>>()?;
        let duration = start.elapsed();
        crate::config::verbose_print(&format!("Mmap search took {:?}", duration));
        Ok(results)
    }

    fn search_lmdb(&self, query_vector: &[Number], _state: &State) -> Result<Vec<(Number, String, usize, Vec<Number>, Metadata)>> {
        let start = Instant::now();
        let entries = self.lmdb.iter()?;
        let results: Result<Vec<_>> = entries
            .par_iter()
            .map(|(key, value)| -> Result<_> {
                let entry: VectorEntry = bincode::deserialize(value)?;
                let mut vector = entry.vector.clone();
                normalize_vector(&mut vector);
                let similarity = compute_cosine_similarity_simd(query_vector, &vector);
                Ok((similarity, key.clone(), 0, vector, entry.metadata))
            })
            .collect();
        let results = results?;
        let duration = start.elapsed();
        crate::config::verbose_print(&format!("LMDB search took {:?}", duration));
        Ok(results)
    }

    pub fn list_entries_lmdb(&self) -> Result<Vec<String>> {
        let rtxn = self.lmdb.env.read_txn()?;
        let iter = self.lmdb.db.iter(&rtxn)?;
        let entries: Result<Vec<_>, _> = iter.map(|res| res.map(|(k, _)| k.to_string())).collect();
        Ok(entries?)
    }

    pub fn migrate_to_lmdb(&self) -> Result<()> {
        for i in 0..self.record_count {
            let label = self.get_label(i)?;
            let vector = self.get_vector(i)?;
            let metadata = self.get_metadata(i)?;
            let entry = VectorEntry {
                label: label.clone(),
                vector,
                metadata,
            };
            self.add_entry_lmdb(&entry)?;
        }
        Ok(())
    }

    pub fn migrate_to_mmap(&self, path: &str) -> Result<()> {
        let entries = self.lmdb.iter()?;
        for (_, value) in entries {
            let entry: VectorEntry = bincode::deserialize(&value)?;
            let chunk = serialize_chunk(&entry, self.label_size)?;
            append_to_db(path, &chunk)?;
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct SearchTimings {
    pub mmap_duration: std::time::Duration,
    pub lmdb_duration: std::time::Duration,
    pub sort_duration: std::time::Duration,
    pub total_duration: std::time::Duration,
}

pub fn parse_input_line(line: &str, state: &State) -> Result<VectorEntry> {
    let entry: VectorEntry = serde_json::from_str(line).context("Failed to parse JSON input")?;

    if entry.vector.len() != state.dimensions {
        anyhow::bail!(
            "Vector dimension mismatch: expected {}, got {}",
            state.dimensions,
            entry.vector.len()
        );
    }

    Ok(entry)
}

pub fn serialize_chunk(entry: &VectorEntry, label_size: usize) -> Result<Vec<u8>> {
    let mut chunk = Vec::with_capacity(entry.vector.len() * size_of::<Number>() + label_size);

    // Serialize vector
    chunk.extend(entry.vector.iter().flat_map(|&num| num.to_le_bytes()));

    // Serialize label with padding
    let label_bytes = entry.label.as_bytes();
    if label_bytes.len() >= label_size {
        anyhow::bail!(
            "Label '{}' is too long (max {} bytes).",
            entry.label,
            label_size - 1
        );
    }
    chunk.extend(label_bytes);
    chunk.resize(entry.vector.len() * size_of::<Number>() + label_size, 0);

    // Serialize metadata
    let metadata_json = serde_json::to_string(&entry.metadata)?;
    let metadata_bytes = metadata_json.as_bytes();
    let metadata_length = metadata_bytes.len() as u32;
    chunk.extend(&metadata_length.to_le_bytes());
    chunk.extend(metadata_bytes);

    Ok(chunk)
}

pub fn append_to_db(path: &str, chunk: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("Failed to open or create database file '{}'", path))?;
    file.write_all(chunk)
        .context("Failed to write to database file")?;
    Ok(())
}