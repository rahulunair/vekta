use crate::config::Number;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Clone)]
pub struct VectorEntry {
    pub label: String,
    pub unique_id: String,
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
