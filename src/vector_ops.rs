use crate::config::{Number, EPSILON};
use wide::f32x8;

/// Compute cosine similarity between two pre-normalized vectors using SIMD operations.
/// Both input vectors `a` and `b` are expected to be normalized before calling this function.
pub fn compute_cosine_similarity_simd(a: &[Number], b: &[Number]) -> Option<Number> {
    if a.len() != b.len() {
        println!("Debug: Vector length mismatch: {} vs {}", a.len(), b.len());
        return None;
    }

    let mut dot_product = f32x8::splat(0.0);
    let mut mag_a = f32x8::splat(0.0);
    let mut mag_b = f32x8::splat(0.0);

    let len = a.len();
    let simd_len = len - (len % 8);

    // SIMD loop
    for i in (0..simd_len).step_by(8) {
        let va = f32x8::new([
            a[i],
            a[i + 1],
            a[i + 2],
            a[i + 3],
            a[i + 4],
            a[i + 5],
            a[i + 6],
            a[i + 7],
        ]);
        let vb = f32x8::new([
            b[i],
            b[i + 1],
            b[i + 2],
            b[i + 3],
            b[i + 4],
            b[i + 5],
            b[i + 6],
            b[i + 7],
        ]);
        dot_product += va * vb;
        mag_a += va * va;
        mag_b += vb * vb;
    }

    let mut scalar_dot_product = dot_product.reduce_add();
    let mut scalar_mag_a = mag_a.reduce_add();
    let mut scalar_mag_b = mag_b.reduce_add();

    // Handle remaining elements
    for i in simd_len..len {
        scalar_dot_product += a[i] * b[i];
        scalar_mag_a += a[i] * a[i];
        scalar_mag_b += b[i] * b[i];
    }

    let denominator = (scalar_mag_a * scalar_mag_b).sqrt();
    if denominator < EPSILON {
        println!("Debug: Denominator too small: {}", denominator);
        Some(0.0)
    } else {
        let similarity = ((scalar_dot_product / denominator).clamp(-1.0, 1.0) + 1.0) / 2.0;
        println!("Debug: Computed similarity: {}", similarity);
        Some(similarity)
    }
}

pub fn normalize_vector(vector: &mut [Number]) {
    let magnitude: Number = vector.iter().map(|&x| x * x).sum::<Number>().sqrt();
    if magnitude > EPSILON {
        for x in vector.iter_mut() {
            *x /= magnitude;
        }
    }
}
