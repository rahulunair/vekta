mod config;
mod database;
mod vector_ops;
mod ann;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::collections::HashSet;
use std::io::{self, BufRead};

use crate::config::State;
use crate::database::{Database, parse_input_line, serialize_chunk, append_to_db};
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
    let mut added_labels = HashSet::new();

    reader.lines().try_for_each(|line_result| -> Result<()> {
        let line = line_result?;
        let mut entry = parse_input_line(&line, state)?;
        if added_labels.contains(&entry.label) {
            eprintln!("Warning: Duplicate label '{}' found. Skipping.", entry.label);
            return Ok(());
        }
        normalize_vector(&mut entry.vector);
        let chunk = serialize_chunk(&entry, state.label_size)?;
        append_to_db(&state.path, &chunk)?;
        added_labels.insert(entry.label.clone());
        config::verbose_print(&format!("Added vector with label '{}'", entry.label));
        Ok(())
    })?;

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
    config::verbose_print("Starting search");
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let query_entry = parse_input_line(&input, state)?;
    let mut query_vector = query_entry.vector;

    normalize_vector(&mut query_vector);

    let db = Database::open(state)?;
    let results = db.search(&query_vector, state)?;

    config::verbose_print(&format!(
        "All similarities: {:?}",
        results.iter().map(|(s, l, _, _, _)| (l, s)).collect::<Vec<_>>()
    ));

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
