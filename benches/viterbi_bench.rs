use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use crfs::{Context, Flag};

fn benchmark_viterbi_by_l(c: &mut Criterion) {
    let mut group = c.benchmark_group("viterbi_by_l");
    
    // Test different L values: 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20
    let l_values = vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20];
    let t = 10; // Sequence length
    
    for l in l_values {
        group.bench_with_input(BenchmarkId::from_parameter(l), &l, |b, &l| {
            let mut ctx = Context::new(Flag::VITERBI, l as u32, t);
            ctx.num_items = t;
            
            // Initialize with some dummy scores
            for i in 0..t as usize {
                for j in 0..l {
                    ctx.state[i * l + j] = (i as f64 * 0.1) + (j as f64 * 0.05);
                }
            }
            
            for i in 0..l {
                for j in 0..l {
                    ctx.trans[i * l + j] = (i as f64 * 0.02) + (j as f64 * 0.01);
                    ctx.trans_t[j * l + i] = ctx.trans[i * l + j];
                }
            }
            
            b.iter(|| {
                let _result = ctx.viterbi();
                black_box(_result);
            });
        });
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_viterbi_by_l);
criterion_main!(benches);
