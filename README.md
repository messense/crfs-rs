# crfs-rs

[![CI](https://github.com/messense/crfs-rs/workflows/CI/badge.svg)](https://github.com/messense/crfs-rs/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/messense/crfs-rs/branch/main/graph/badge.svg)](https://codecov.io/gh/messense/crfs-rs)
[![Crates.io](https://img.shields.io/crates/v/crfs.svg)](https://crates.io/crates/crfs)
[![docs.rs](https://docs.rs/crfs/badge.svg)](https://docs.rs/crfs/)

Pure Rust port of CRFsuite: a fast implementation of Conditional Random Fields (CRFs)

## Installation

Add it to your ``Cargo.toml``:

```toml
[dependencies]
crfs = "0.2"
```

## Performance

Performance comparsion with CRFsuite on MacBook Pro (13-inch, M4 MAX, 2024)

```bash
$ cargo bench --bench crf_bench -- --output-format bencher
test tag/crfs ... bench:         579 ns/iter (+/- 15)
test tag/crfsuite ... bench:        1185 ns/iter (+/- 20)
```

Last updated on 2026-02-01.

## License

This work is released under the MIT license. A copy of the license is provided
in the [LICENSE](./LICENSE) file.
