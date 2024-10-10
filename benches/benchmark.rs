use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::prelude::*;
use serde_json::Value;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::collections::HashSet;

fn configure_criterion() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(70))
        .warm_up_time(std::time::Duration::from_secs(10))
        .configure_from_args()
}

fn load_embeddings(c: &mut Criterion) {
    let embedding_file = "embeddings.jsonl";

    if !std::path::Path::new(embedding_file).exists() {
        panic!("embeddings.jsonl not found. Run the Python script first.");
    }

    env::set_var("VEKTA_PATH", "benchmark_db.bin");
    env::set_var("VEKTA_DIMENSIONS", "384");
    env::set_var("VEKTA_LABEL_SIZE", "32");
    env::set_var("VEKTA_TOP_K", "10");
    env::set_var("VEKTA_ANN_NUM_PROJECTIONS", "20");

    c.bench_function("load embeddings", |b| {
        b.iter(|| {
            let file = File::open(embedding_file).unwrap();
            let reader = BufReader::new(file);
            let mut added_labels = std::collections::HashSet::new();

            for line in reader.lines() {
                let line = line.unwrap();
                let v: Value = serde_json::from_str(&line).unwrap();
                let label = v["label"].as_str().unwrap();

                if added_labels.contains(label) {
                    continue;
                }

                let mut child = Command::new("./target/release/vekta")
                    .arg("add")
                    .stdin(Stdio::piped())
                    .spawn()
                    .expect("Failed to spawn command");

                let mut stdin = child.stdin.take().expect("Failed to open stdin");
                stdin
                    .write_all(line.as_bytes())
                    .expect("Failed to write to stdin");
                drop(stdin);

                let output = child.wait_with_output().expect("Failed to read stdout");
                assert!(output.status.success());

                added_labels.insert(label.to_string());
            }
        })
    });
}

fn search_embeddings(c: &mut Criterion) {
    if !std::path::Path::new("benchmark_db.bin").exists() {
        panic!("benchmark_db.bin not found. Run the load embeddings benchmark first.");
    }

    env::set_var("VEKTA_PATH", "benchmark_db.bin");
    env::set_var("VEKTA_DIMENSIONS", "384");
    env::set_var("VEKTA_LABEL_SIZE", "32");
    env::set_var("VEKTA_TOP_K", "10");
    env::set_var("VEKTA_ANN_NUM_PROJECTIONS", "20");

    let file = File::open("embeddings.jsonl").unwrap();
    let reader = BufReader::new(file);
    let embeddings: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();

    let mut rng = rand::thread_rng();

    let mut group = c.benchmark_group("search_embeddings");
    for search_method in ["exact", "ann"] {
        group.bench_with_input(BenchmarkId::new("search", search_method), &search_method, |b, &search_method| {
            b.iter(|| {
                let query = embeddings.choose(&mut rng).unwrap();
                env::set_var("VEKTA_SEARCH_METHOD", search_method);
                run_search(query)
            })
        });
    }
    group.finish();

    c.bench_function("search embeddings (exact vs ANN comparison)", |b| {
        b.iter(|| {
            let query = embeddings.choose(&mut rng).unwrap();

            env::set_var("VEKTA_SEARCH_METHOD", "exact");
            let exact_results = run_search(query);

            env::set_var("VEKTA_SEARCH_METHOD", "ann");
            let ann_results = run_search(query);

            compare_results(&exact_results, &ann_results);
        })
    });
}

fn run_search(query: &str) -> String {
    let mut child = Command::new("./target/release/vekta")
        .arg("search")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to spawn command");

    let mut stdin = child.stdin.take().expect("Failed to open stdin");
    stdin
        .write_all(query.as_bytes())
        .expect("Failed to write to stdin");
    drop(stdin);

    let output = child.wait_with_output().expect("Failed to read stdout");
    assert!(
        output.status.success(),
        "Command failed with exit code: {:?}",
        output.status.code()
    );

    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn compare_results(exact_results: &str, ann_results: &str) {
    // Parse JSON results
    for (method, results) in [("Exact", exact_results), ("ANN", ann_results)] {
        let json: serde_json::Value = serde_json::from_str(results).expect(&format!("Failed to parse {} JSON", method));
        
        assert!(json.get("query_label").is_some(), "{} search: Output doesn't contain 'query_label'", method);
        assert!(json.get("query_vector").is_some(), "{} search: Output doesn't contain 'query_vector'", method);
        assert!(json.get("database_record_count").is_some(), "{} search: Output doesn't contain 'database_record_count'", method);
        assert!(json.get("results").is_some(), "{} search: Output doesn't contain 'results'", method);
        assert!(json.get("actual_results_count").is_some(), "{} search: Output doesn't contain 'actual_results_count'", method);
        assert!(json.get("requested_results_count").is_some(), "{} search: Output doesn't contain 'requested_results_count'", method);
    }

    // Extract and compare top results
    let exact_top = extract_top_results(exact_results);
    let ann_top = extract_top_results(ann_results);

    let overlap = exact_top.intersection(&ann_top).count();
    let overlap_percentage = (overlap as f64 / exact_top.len() as f64) * 100.0;

    println!("Overlap between exact and ANN results: {:.2}%", overlap_percentage);
}

fn extract_top_results(results: &str) -> HashSet<String> {
    let json: serde_json::Value = serde_json::from_str(results).expect("Failed to parse JSON");
    json["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|result| result["label"].as_str().unwrap().to_string())
        .collect()
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = load_embeddings, search_embeddings
}
criterion_main!(benches);