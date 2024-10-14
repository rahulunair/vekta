use crate::config::{Number, EPSILON};
use wide::f32x8;

pub fn compute_cosine_similarity_simd(a: &[Number], b: &[Number]) -> Number {
    let mut dot_product = f32x8::splat(0.0);
    let mut mag_a = f32x8::splat(0.0);
    let mut mag_b = f32x8::splat(0.0);

    let len = a.len();
    let mut i = 0;

    while i + 8 <= len {
        let va = f32x8::new([a[i], a[i+1], a[i+2], a[i+3], a[i+4], a[i+5], a[i+6], a[i+7]]);
        let vb = f32x8::new([b[i], b[i+1], b[i+2], b[i+3], b[i+4], b[i+5], b[i+6], b[i+7]]);
        dot_product += va * vb;
        mag_a += va * va;
        mag_b += vb * vb;
        i += 8;
    }

    let mut scalar_dot_product = dot_product.reduce_add();
    let mut scalar_mag_a = mag_a.reduce_add();
    let mut scalar_mag_b = mag_b.reduce_add();

    // Handle the remaining elements
    for j in i..len {
        scalar_dot_product += a[j] * b[j];
        scalar_mag_a += a[j] * a[j];
        scalar_mag_b += b[j] * b[j];
    }

    let similarity = scalar_dot_product / (scalar_mag_a.sqrt() * scalar_mag_b.sqrt());
    similarity.clamp(-1.0, 1.0)
}

pub fn normalize_vector(vector: &mut [Number]) {
    let magnitude: Number = vector.iter().map(|&x| x * x).sum::<Number>().sqrt();
    if magnitude > EPSILON {
        for x in vector.iter_mut() {
            *x /= magnitude;
        }
    }
}