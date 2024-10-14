use anyhow::{Context, Result};
use memmap2::Mmap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::mem::size_of;
use std::sync::Arc;

use crate::ann::RandomProjectionIndex;
use crate::config::{Number, State};
use crate::vector_ops::{compute_cosine_similarity_simd, normalize_vector};

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

pub struct Database {
    mmap: Arc<Mmap>,
    pub record_count: usize,
    vector_size: usize,
    label_size: usize,
    ann_index: Option<RandomProjectionIndex>,
    offsets: Vec<usize>,
}

impl Database {
    pub fn open(state: &State) -> Result<Self> {
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

        Ok(Self {
            mmap,
            record_count,
            vector_size: state.vector_size,
            label_size: state.label_size,
            ann_index,
            offsets,
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

    pub fn search(&self, query_vector: &[Number], state: &State) -> Result<Vec<(Number, String, usize, Vec<Number>, Metadata)>> {
        let search_range = if let Some(ann_index) = self.ann_index.as_ref() {
            crate::config::verbose_print("Using ANN search method");
            ann_index.search(query_vector, state.top_k * 50)
        } else {
            crate::config::verbose_print("Using exact search method");
            (0..self.record_count).collect()
        };

        search_range
            .into_par_iter()
            .map(|i| -> Result<_> {
                let mut vector = self.get_vector(i)?;
                let label = self.get_label(i)?;
                let metadata = self.get_metadata(i)?;
                normalize_vector(&mut vector);
                let similarity = compute_cosine_similarity_simd(query_vector, &vector);
                Ok((similarity, label, i, vector, metadata))
            })
            .collect()
    }
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
