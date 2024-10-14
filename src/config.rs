use anyhow::{Context, Result};
use config::{Config, ConfigError, Environment, File as ConfigFile};
use serde::Deserialize;
use std::env;
use std::mem::size_of;

pub type Number = f32;

pub const EPSILON: f32 = 1e-6;

#[derive(Deserialize)]
pub struct VektaConfig {
    pub path: Option<String>,
    pub dimensions: Option<usize>,
    pub label_size: Option<usize>,
    pub top_k: Option<usize>,
    pub search_method: Option<String>,
    pub ann_num_projections: Option<usize>,
}

impl VektaConfig {
    pub fn try_from(config: &Config) -> Result<Self, ConfigError> {
        Ok(VektaConfig {
            path: config.get("path").ok(),
            dimensions: config.get("dimensions").ok(),
            label_size: config.get("label_size").ok(),
            top_k: config.get("top_k").ok(),
            search_method: config.get("search_method").ok(),
            ann_num_projections: config.get("ann_num_projections").ok(),
        })
    }
}

pub struct State {
    pub path: String,
    pub dimensions: usize,
    pub label_size: usize,
    pub vector_size: usize,
    pub chunk_size: usize,
    pub top_k: usize,
    pub search_method: String,
    pub ann_num_projections: usize,
}

impl State {
    pub fn new() -> Result<Self> {
        let mut config = Config::default();
        #[allow(deprecated)]
        {
            config.merge(ConfigFile::with_name("vekta_config").required(false))?;
            config.merge(Environment::with_prefix("VEKTA"))?;
        }

        let vekta_config = VektaConfig::try_from(&config)?;

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

    pub fn print_config(&self) {
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

pub fn verbose_print(message: &str) {
    if env::var("VEKTA_VERBOSE").unwrap_or_else(|_| "false".to_string()) == "true" {
        eprintln!("{}", message);
    }
}