mod ann;
mod config;
mod database;
mod search;
mod vector_entry;
mod vector_ops;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::collections::HashSet;
use std::io::{self, BufRead};

use crate::config::State;
use crate::database::{parse_input_line, VectorDatabase};
use crate::search::SearchEngine;
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
}

fn add_command(state: &State) -> Result<()> {
    let stdin = io::stdin();
    let reader = stdin.lock();
    let mut db = VectorDatabase::open(state)?;
    let mut added_labels = HashSet::new();

    for (i, line_result) in reader.lines().enumerate() {
        let line = line_result.context("Failed to read input line")?;
        println!("Processing line {}: {}", i, line);
        let mut entry = parse_input_line(&line, state)
            .with_context(|| format!("Failed to parse input line: {}", line))?;

        if db.label_exists(&entry.label)? {
            eprintln!(
                "Warning: Label '{}' already exists in the database. Skipping.",
                entry.label
            );
            continue;
        }

        if added_labels.contains(&entry.label) {
            eprintln!(
                "Warning: Duplicate label '{}' found in current input. Skipping.",
                entry.label
            );
            continue;
        }

        normalize_vector(&mut entry.vector);

        let result = db
            .add_entry(&entry) // Pass a reference to entry
            .with_context(|| format!("Failed to add entry with label: {}", entry.label))?;

        added_labels.insert(entry.label.clone());
        config::verbose_print(&format!("Added vector with label '{}'", result));
    }

    Ok(())
}

fn list_command(state: &State) -> Result<()> {
    let db = VectorDatabase::open(state)?;

    println!("LMDB entries:");
    let lmdb_entries = db.list_entries()?;
    for entry in lmdb_entries {
        println!("{}", entry);
    }

    Ok(())
}

fn search_command(state: &State) -> Result<()> {
    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .context("Failed to read input")?;

    if input.trim().is_empty() {
        return Err(anyhow::anyhow!(
            "Error: No input provided for the search query."
        ));
    }

    let query_entry =
        parse_input_line(&input, state).context("Failed to parse input as a valid query")?;
    let query_vector = &query_entry.vector;

    let db = VectorDatabase::open(state)?;
    println!("Debug: Database opened, record count: {}", db.count()?);

    let search_engine = SearchEngine::new(db, state)?;
    let (results, timings) = search_engine.search(query_vector, state)?;

    let output = serde_json::json!({
        "query": {
            "label": query_entry.label,
            "unique_id": query_entry.unique_id,
            "vector": &query_vector[..5.min(query_vector.len())],
            "metadata": query_entry.metadata,
        },
        "database_record_count": search_engine.db.count()?,
        "results": results.iter().map(|result| {
            serde_json::json!({
                "label": result.label,
                "unique_id": result.unique_id,
                "similarity": result.similarity,
                "metadata": result.metadata,
            })
        }).collect::<Vec<_>>(),
        "actual_results_count": results.len(),
        "requested_results_count": state.top_k,
        "timings": {
            "search_duration_ms": timings.search_duration.as_millis(),
            "sort_duration_ms": timings.sort_duration.as_millis(),
            "total_duration_ms": timings.total_duration.as_millis(),
        }
    });

    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
}

fn config_command(state: &State) -> Result<()> {
    state.print_config();
    Ok(())
}

fn main() -> Result<()> {
    let args = Cli::parse();
    let state = State::new()?;

    let result = match &args.command {
        Commands::Add => add_command(&state),
        Commands::List => list_command(&state),
        Commands::Search => search_command(&state),
        Commands::Config => config_command(&state),
    };

    if let Err(e) = result {
        eprintln!("Error: {:?}", e);
        std::process::exit(1);
    }

    Ok(())
}
