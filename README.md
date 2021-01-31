# crfs-rs

[![CI](https://github.com/messense/crfs-rs/workflows/CI/badge.svg)](https://github.com/messense/crfs-rs/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/messense/crfs-rs/branch/master/graph/badge.svg)](https://codecov.io/gh/messense/crfs-rs)
[![Crates.io](https://img.shields.io/crates/v/crfs.svg)](https://crates.io/crates/crfs)
[![docs.rs](https://docs.rs/crfs/badge.svg)](https://docs.rs/crfs/)

Pure Rust port of CRFsuite: a fast implementation of Conditional Random Fields (CRFs)

**Currently only support prediction, model training is not supported.**
For training you can use [crfsuite-rs](https://github.com/messense/crfsuite-rs).

## Installation

Add it to your ``Cargo.toml``:

```toml
[dependencies]
crfs = "0.1"
```

## License

This work is released under the MIT license. A copy of the license is provided
in the [LICENSE](./LICENSE) file.