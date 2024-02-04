use bencher::{benchmark_group, benchmark_main, black_box, Bencher};
use rand::prelude::*;
use sobol_burley::{sample, sample_8d};

//----

fn gen_1000_samples_8d(bench: &mut Bencher) {
    bench.iter(|| {
        for i in 0..250u32 {
            black_box(sample_8d(i, 0, 1234567890));
        }
    });
}

fn gen_1000_samples_incoherent_8d(bench: &mut Bencher) {
    let mut rng = rand::thread_rng();
    bench.iter(|| {
        let s = rng.gen::<u32>();
        let d = rng.gen::<u32>();
        let seed = rng.gen::<u32>();
        for i in 0..250u32 {
            black_box(sample_8d(
                s.wrapping_add(i).wrapping_mul(512),
                d.wrapping_add(i).wrapping_mul(97) % 32,
                seed,
            ));
        }
    });
}

fn gen_1000_samples(bench: &mut Bencher) {
    bench.iter(|| {
        for i in 0..1000u32 {
            black_box(sample(i, 0, 1234567890));
        }
    });
}

fn gen_1000_samples_incoherent(bench: &mut Bencher) {
    let mut rng = rand::thread_rng();
    bench.iter(|| {
        let s = rng.gen::<u32>();
        let d = rng.gen::<u32>();
        let seed = rng.gen::<u32>();
        for i in 0..1000u32 {
            black_box(sample(
                s.wrapping_add(i).wrapping_mul(512),
                d.wrapping_add(i).wrapping_mul(97) % 128,
                seed,
            ));
        }
    });
}

//----

benchmark_group!(
    benches,
    gen_1000_samples,
    gen_1000_samples_incoherent,
    gen_1000_samples_8d,
    gen_1000_samples_incoherent_8d,
);
benchmark_main!(benches);
