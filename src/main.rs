mod config;
mod database;
mod vector_ops;
mod ann;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::collections::HashSet;
use std::io::{self, BufRead};

use crate::config::State;
use crate::database::{VectorDatabase, parse_input_line, serialize_chunk, append_to_db};
use crate::vector_ops::normalize_vector;

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
    Migrate { to: String }, // New command to migrate data
}

fn add_command(state: &State) -> Result<()> {
    let stdin = io::stdin();
    let reader = stdin.lock();
    let mut added_labels = HashSet::new();
    let mut db = VectorDatabase::open(state)?;

    reader.lines().try_for_each(|line_result| -> Result<()> {
        let line = line_result?;
        let mut entry = parse_input_line(&line, state)?;
        if added_labels.contains(&entry.label) {
            eprintln!("Warning: Duplicate label '{}' found. Skipping.", entry.label);
            return Ok(());
        }
        normalize_vector(&mut entry.vector);
        
        // Add to memory-mapped file
        let chunk = serialize_chunk(&entry, state.label_size)?;
        append_to_db(&state.path, &chunk)?;
        
        // Add to LMDB
        db.add_entry_lmdb(&entry)?;
        
        added_labels.insert(entry.label.clone());
        config::verbose_print(&format!("Added vector with label '{}'", entry.label));
        Ok(())
    })?;

    // Update ANN index
    for i in 0..db.record_count {
        let vector = db.get_vector(i)?;
        db.add_to_ann(&vector);
    }

    Ok(())
}

fn list_command(state: &State) -> Result<()> {
    let db = VectorDatabase::open(state)?;
    
    println!("Memory-mapped file entries:");
    for i in 0..db.record_count {
        let label = db.get_label(i)?;
        println!("{}", label);
    }
    
    println!("\nLMDB entries:");
    let lmdb_entries = db.list_entries_lmdb()?;
    for entry in lmdb_entries {
        println!("{}", entry);
    }
    
    Ok(())
}

fn search_command(state: &State) -> Result<()> {
    config::verbose_print("Starting search");
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let query_entry = parse_input_line(&input, state)?;
    let mut query_vector = query_entry.vector;

    normalize_vector(&mut query_vector);

    let db = VectorDatabase::open(state)?;
    let (results, timings) = db.search(&query_vector, &state)?;

    println!("Search timings:");
    println!("  Mmap search: {:?}", timings.mmap_duration);
    println!("  LMDB search: {:?}", timings.lmdb_duration);
    println!("  Sort duration: {:?}", timings.sort_duration);
    println!("  Total duration: {:?}", timings.total_duration);

    let output = serde_json::json!({
        "query": {
            "label": query_entry.label,
            "vector": &query_vector[..5.min(query_vector.len())],
            "metadata": query_entry.metadata,
        },
        "database_record_count": db.record_count,
        "results": results.iter().map(|(similarity, label, _, _, metadata)| {
            serde_json::json!({
                "label": label,
                "similarity": similarity,
                "metadata": metadata,
            })
        }).collect::<Vec<_>>(),
        "actual_results_count": results.len(),
        "requested_results_count": state.top_k
    });

    println!("{}", serde_json::to_string(&output)?);

    Ok(())
}

fn config_command(state: &State) -> Result<()> {
    state.print_config();
    Ok(())
}

fn migrate_command(state: &State, to: &str) -> Result<()> {
    let db = VectorDatabase::open(state)?;
    match to {
        "lmdb" => db.migrate_to_lmdb()?,
        "mmap" => db.migrate_to_mmap(&state.path)?,
        _ => anyhow::bail!("Invalid migration target. Use 'lmdb' or 'mmap'."),
    }
    println!("Migration completed successfully.");
    Ok(())
}

fn main() -> Result<()> {
    let args = Cli::parse();
    let state = State::new()?;

    match &args.command {
        Commands::Add => add_command(&state)?,
        Commands::List => list_command(&state)?,
        Commands::Search => search_command(&state)?,
        Commands::Config => config_command(&state)?,
        Commands::Migrate { to } => migrate_command(&state, to)?,
    }
    Ok(())
}
