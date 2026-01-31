use std::fs;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("tag");
    group.bench_function("crfs", |b| {
        use crfs::Attribute;

        let buf = fs::read("tests/model.crfsuite").unwrap();
        let model = crfs::Model::new(&buf).unwrap();
        let xseq = vec![
            vec![Attribute::new("walk", 1.0), Attribute::new("shop", 0.5)],
            vec![Attribute::new("walk", 1.0)],
            vec![Attribute::new("walk", 1.0), Attribute::new("clean", 0.5)],
            vec![Attribute::new("shop", 0.5), Attribute::new("clean", 0.5)],
            vec![Attribute::new("walk", 0.5), Attribute::new("clean", 1.0)],
            vec![Attribute::new("clean", 1.0), Attribute::new("shop", 0.1)],
            vec![Attribute::new("walk", 1.0), Attribute::new("shop", 0.5)],
            vec![],
            vec![Attribute::new("clean", 1.0)],
        ];
        b.iter(|| {
            let tagger = model.tagger().unwrap();
            let _res = tagger.tag(black_box(&xseq)).unwrap();
        })
    });
    group.bench_function("crfsuite", |b| {
        use crfsuite::Attribute;

        let buf = fs::read("tests/model.crfsuite").unwrap();
        let model = crfsuite::Model::from_memory(&buf).unwrap();
        let xseq = vec![
            vec![Attribute::new("walk", 1.0), Attribute::new("shop", 0.5)],
            vec![Attribute::new("walk", 1.0)],
            vec![Attribute::new("walk", 1.0), Attribute::new("clean", 0.5)],
            vec![Attribute::new("shop", 0.5), Attribute::new("clean", 0.5)],
            vec![Attribute::new("walk", 0.5), Attribute::new("clean", 1.0)],
            vec![Attribute::new("clean", 1.0), Attribute::new("shop", 0.1)],
            vec![Attribute::new("walk", 1.0), Attribute::new("shop", 0.5)],
            vec![],
            vec![Attribute::new("clean", 1.0)],
        ];
        b.iter(|| {
            let mut tagger = model.tagger().unwrap();
            let _res = tagger.tag(black_box(&xseq)).unwrap();
        })
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
