use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
use serde_json::Value;
use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

const NUM_ELEMENTS: usize = 100; // Number of elements to add
const NUM_QUERIES: usize = 10; // Number of queries to perform

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

    env::set_var("VEKTA_PATH", "benchmark_db");
    env::set_var("VEKTA_DIMENSIONS", "384");
    env::set_var("VEKTA_LABEL_SIZE", "64");
    env::set_var("VEKTA_TOP_K", "10");
    env::set_var("VEKTA_SEARCH_METHOD", "exact");
    env::set_var("VEKTA_SIMILARITY_THRESHOLD", "0.0");
    env::set_var("VEKTA_VERBOSE", "true");

    c.bench_function(&format!("load {} embeddings", NUM_ELEMENTS), |b| {
        b.iter(|| {
            let file = File::open(embedding_file).unwrap();
            let reader = BufReader::new(file);

            for (i, line) in reader.lines().enumerate() {
                if i >= NUM_ELEMENTS {
                    break;
                }
                let line = line.unwrap();
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
            }
        })
    });
}

fn search_embeddings(c: &mut Criterion) {
    if !std::path::Path::new("benchmark_db").exists() {
        panic!("benchmark_db directory not found. Run the load embeddings benchmark first.");
    }

    let query_file = "query_embeddings.jsonl";
    if !std::path::Path::new(query_file).exists() {
        panic!("query_embeddings.jsonl not found. Create this file with query vectors.");
    }

    env::set_var("VEKTA_PATH", "benchmark_db");
    env::set_var("VEKTA_DIMENSIONS", "384");
    env::set_var("VEKTA_LABEL_SIZE", "64");
    env::set_var("VEKTA_TOP_K", "10");
    env::set_var("VEKTA_SIMILARITY_THRESHOLD", "0.0");
    env::set_var("VEKTA_VERBOSE", "true");

    let file = File::open(query_file).unwrap();
    let reader = BufReader::new(file);
    let query_embeddings: Vec<String> = reader.lines().take(NUM_QUERIES).map(|l| l.unwrap()).collect();

    let mut group = c.benchmark_group("search_embeddings");
    for search_method in ["exact", "ann", "hybrid"] {
        group.bench_with_input(
            BenchmarkId::new(&format!("search_{}_queries", NUM_QUERIES), search_method),
            &search_method,
            |b, &search_method| {
                b.iter(|| {
                    env::set_var("VEKTA_SEARCH_METHOD", search_method);
                    for query in &query_embeddings {
                        run_search(query);
                    }
                })
            },
        );
    }
    group.finish();

    c.bench_function(&format!("compare_{}_searches", NUM_QUERIES), |b| {
        b.iter(|| {
            for query in &query_embeddings {
                env::set_var("VEKTA_SEARCH_METHOD", "exact");
                let exact_results = run_search(query);

                env::set_var("VEKTA_SEARCH_METHOD", "ann");
                let ann_results = run_search(query);

                env::set_var("VEKTA_SEARCH_METHOD", "hybrid");
                let hybrid_results = run_search(query);

                compare_results(&exact_results, &ann_results, &hybrid_results);
            }
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

fn compare_results(exact_results: &str, ann_results: &str, hybrid_results: &str) {
    for (method, results) in [
        ("Exact", exact_results),
        ("ANN", ann_results),
        ("Hybrid", hybrid_results),
    ] {
        let json: serde_json::Value =
            serde_json::from_str(results).expect(&format!("Failed to parse {} JSON", method));

        assert!(
            json["query"]["label"].is_string(),
            "{} search: Output doesn't contain 'query.label'",
            method
        );
        assert!(
            json["query"]["vector"].is_array(),
            "{} search: Output doesn't contain 'query.vector'",
            method
        );
        assert!(
            json["database_record_count"].is_number(),
            "{} search: Output doesn't contain 'database_record_count'",
            method
        );
        assert!(
            json["results"].is_array(),
            "{} search: Output doesn't contain 'results'",
            method
        );
        assert!(
            json["actual_results_count"].is_number(),
            "{} search: Output doesn't contain 'actual_results_count'",
            method
        );
        assert!(
            json["requested_results_count"].is_number(),
            "{} search: Output doesn't contain 'requested_results_count'",
            method
        );
    }

    let exact_top = extract_top_results(exact_results);
    let ann_top = extract_top_results(ann_results);
    let hybrid_top = extract_top_results(hybrid_results);

    let ann_overlap = exact_top.intersection(&ann_top).count();
    let hybrid_overlap = exact_top.intersection(&hybrid_top).count();

    let ann_overlap_percentage = (ann_overlap as f64 / exact_top.len() as f64) * 100.0;
    let hybrid_overlap_percentage = (hybrid_overlap as f64 / exact_top.len() as f64) * 100.0;

    println!(
        "Overlap between exact and ANN results: {:.2}%",
        ann_overlap_percentage
    );
    println!(
        "Overlap between exact and hybrid results: {:.2}%",
        hybrid_overlap_percentage
    );
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
