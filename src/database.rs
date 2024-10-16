use anyhow::{Context, Result};
use bincode;
use heed::types::*;
use heed::EnvOpenOptions;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

use crate::config::{Number, State};
use crate::vector_entry::{Metadata, VectorEntry};
use crate::vector_ops::normalize_vector;

pub struct LmdbWrapper {
    env: heed::Env,
    db: heed::Database<Str, SerdeBincode<Vec<u8>>>,
    label_index: heed::Database<Str, Str>,
}

impl LmdbWrapper {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = PathBuf::from(path.as_ref());

        fs::create_dir_all(&path).with_context(|| {
            format!(
                "Failed to create directory for LMDB at '{}'",
                path.display()
            )
        })?;

        println!("Attempting to open LMDB at: {}", path.display());

        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(10 * 1024 * 1024 * 1024) // 10GB
                .max_dbs(2)
                .open(&path)
                .with_context(|| {
                    format!("Failed to open LMDB environment at '{}'", path.display())
                })?
        };

        let mut wtxn = env
            .write_txn()
            .with_context(|| "Failed to create write transaction for LMDB")?;
        let db: heed::Database<Str, SerdeBincode<Vec<u8>>> = env
            .create_database(&mut wtxn, Some("main"))
            .with_context(|| "Failed to create main LMDB database")?;
        let label_index: heed::Database<Str, Str> = env
            .create_database(&mut wtxn, Some("label_index"))
            .with_context(|| "Failed to create label index LMDB database")?;
        wtxn.commit()
            .with_context(|| "Failed to commit initial LMDB transaction")?;

        Ok(Self {
            env,
            db,
            label_index,
        })
    }

    pub fn add(&self, key: &str, value: &[u8], label: &str) -> Result<()> {
        let mut wtxn = self.env.write_txn()?;
        self.db.put(&mut wtxn, key, &value.to_vec())?;
        self.label_index.put(&mut wtxn, label, key)?;
        wtxn.commit()?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let rtxn = self.env.read_txn()?;
        Ok(self.db.get(&rtxn, key)?.map(|v| v.to_vec()))
    }

    pub fn label_exists(&self, label: &str) -> Result<bool> {
        let rtxn = self.env.read_txn()?;
        Ok(self.label_index.get(&rtxn, label)?.is_some())
    }

    pub fn iter(&self) -> Result<Vec<(String, Vec<u8>)>> {
        let rtxn = self.env.read_txn()?;
        let iter_result: Vec<(String, Vec<u8>)> = self
            .db
            .iter(&rtxn)?
            .map(|result| result.map(|(k, v)| (k.to_string(), v.to_vec())))
            .collect::<std::result::Result<Vec<_>, heed::Error>>()?;

        Ok(iter_result)
    }

    pub fn count(&self) -> Result<usize> {
        let rtxn = self.env.read_txn()?;
        Ok(self.db.len(&rtxn)?.try_into().unwrap())
    }
}

pub struct VectorDatabase {
    lmdb: LmdbWrapper,
    vector_size: usize,
    label_size: usize,
}

impl VectorDatabase {
    pub fn open(state: &State) -> Result<Self> {
        println!("Opening database at path: {}", state.path);
        let lmdb = LmdbWrapper::new(&state.path)
            .with_context(|| format!("Failed to create LMDB at '{}'", state.path))?;

        Ok(Self {
            lmdb,
            vector_size: state.vector_size,
            label_size: state.label_size,
        })
    }

    pub fn add_entry(&mut self, entry: &VectorEntry) -> Result<String> {
        if entry.vector.len() * std::mem::size_of::<Number>() != self.vector_size {
            anyhow::bail!(
                "Vector size mismatch: expected {} bytes, got {} bytes",
                self.vector_size,
                entry.vector.len() * std::mem::size_of::<Number>()
            );
        }
        if entry.label.len() > self.label_size {
            anyhow::bail!(
                "Label size too large: maximum {} bytes, got {} bytes",
                self.label_size,
                entry.label.len()
            );
        }

        let content_hash = generate_content_hash(entry);
        let mut final_label = entry.label.clone();
        let mut counter = 0;

        loop {
            if let Some(existing_entry) = self.get_entry_by_label(&final_label)? {
                let existing_hash = generate_content_hash(&existing_entry);

                if existing_hash == content_hash {
                    // Content is the same, no need to add
                    return Ok(final_label);
                } else {
                    // Same label, different content. Modify the label
                    counter += 1;
                    final_label = format!("{}_{}", entry.label, counter);
                    continue;
                }
            }

            // Label doesn't exist, we can add the entry
            break;
        }

        // Create a copy of the entry with a normalized vector and possibly modified label
        let mut normalized_entry = entry.clone();
        normalize_vector(&mut normalized_entry.vector);
        normalized_entry.label = final_label.clone();

        let value = bincode::serialize(&normalized_entry)?;
        self.lmdb
            .add(&normalized_entry.unique_id, &value, &final_label)?;
        Ok(final_label)
    }

    #[allow(dead_code)]
    pub fn get_entry(&self, unique_id: &str) -> Result<Option<VectorEntry>> {
        if let Some(value) = self.lmdb.get(unique_id)? {
            Ok(Some(bincode::deserialize(&value)?))
        } else {
            Ok(None)
        }
    }

    pub fn get_entry_by_index(&self, index: usize) -> Result<Option<VectorEntry>> {
        let entries = self.lmdb.iter()?;
        entries
            .get(index)
            .map(|(_, value)| bincode::deserialize(value))
            .transpose()
            .map_err(|e| e.into())
    }

    pub fn list_entries(&self) -> Result<Vec<String>> {
        self.lmdb
            .iter()?
            .into_iter()
            .map(|(key, _)| Ok(key))
            .collect()
    }

    pub fn count(&self) -> Result<usize> {
        self.lmdb.count()
    }

    pub fn label_exists(&self, label: &str) -> Result<bool> {
        self.lmdb.label_exists(label)
    }

    pub fn get_entry_by_label(&self, label: &str) -> Result<Option<VectorEntry>> {
        let rtxn = self.lmdb.env.read_txn()?;
        if let Some(unique_id) = self.lmdb.label_index.get(&rtxn, label)? {
            self.get_entry(&unique_id)
        } else {
            Ok(None)
        }
    }
}

pub fn parse_input_line(line: &str, state: &State) -> Result<VectorEntry> {
    let mut json_value: Value = serde_json::from_str(line)
        .with_context(|| format!("Failed to parse JSON from input line: {}", line))?;

    println!("JSON parsed successfully");

    ensure_utf8(&mut json_value);

    if json_value.get("unique_id").is_none() {
        let label = json_value["label"].as_str().unwrap_or("");
        let vector = json_value["vector"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Vector field is missing or not an array"))?;
        let metadata = json_value["metadata"].clone();

        let mut vector_numbers: Vec<Number> = vector
            .iter()
            .map(|v| {
                v.as_f64()
                    .ok_or_else(|| anyhow::anyhow!("Invalid vector element"))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .map(|f| f as Number)
            .collect();

        // Normalize the vector
        normalize_vector(&mut vector_numbers);

        let metadata: Metadata =
            serde_json::from_value(metadata).with_context(|| "Failed to parse metadata")?;

        let unique_id = generate_unique_id(label, &vector_numbers, &metadata);
        json_value["unique_id"] = Value::String(unique_id);
    }

    let entry: VectorEntry = serde_json::from_value(json_value.clone()).with_context(|| {
        format!(
            "Failed to deserialize JSON into VectorEntry. JSON: {:?}",
            json_value
        )
    })?;

    println!("Deserialized into VectorEntry successfully");

    if entry.vector.len() != state.dimensions {
        anyhow::bail!(
            "Vector dimension mismatch: expected {}, got {}",
            state.dimensions,
            entry.vector.len()
        );
    }

    Ok(entry)
}

pub fn ensure_utf8(value: &mut Value) {
    match value {
        Value::String(s) => {
            if let Err(_) = String::from_utf8(s.clone().into_bytes()) {
                *s = String::from_utf8_lossy(s.as_bytes()).into_owned();
            }
        }
        Value::Array(arr) => {
            for item in arr {
                ensure_utf8(item);
            }
        }
        Value::Object(obj) => {
            for (_, v) in obj {
                ensure_utf8(v);
            }
        }
        _ => {}
    }
}

pub fn generate_unique_id(label: &str, vector: &[Number], metadata: &Metadata) -> String {
    let mut hasher = Sha256::new();
    hasher.update(label.as_bytes());
    for &num in vector {
        hasher.update(num.to_le_bytes());
    }
    hasher.update(serde_json::to_string(metadata).unwrap().as_bytes());
    format!("{:x}", hasher.finalize())
}

fn generate_content_hash(entry: &VectorEntry) -> String {
    let mut hasher = Sha256::new();
    for &num in &entry.vector {
        hasher.update(num.to_le_bytes());
    }
    hasher.update(serde_json::to_string(&entry.metadata).unwrap().as_bytes());
    format!("{:x}", hasher.finalize())
}
