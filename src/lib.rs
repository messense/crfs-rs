//! Pure Rust implementation of Conditional Random Fields (CRF)
//!
//! This library provides both training and prediction capabilities for linear-chain CRFs.
//!
//! # Examples
//!
//! ## Training
//!
//! ```no_run
//! use crfs::train::Trainer;
//! use crfs::Attribute;
//! use std::path::Path;
//!
//! let mut trainer = Trainer::lbfgs();
//! trainer.verbose(true);
//!
//! let xseq = vec![
//!     vec![Attribute::new("walk", 1.0)],
//!     vec![Attribute::new("shop", 1.0)],
//! ];
//! let yseq = vec!["sunny", "rainy"];
//! trainer.append(&xseq, &yseq)?;
//!
//! trainer.params_mut().set_c2(1.0)?;
//! trainer.train(Path::new("model.crfsuite"))?;
//! # Ok::<(), std::io::Error>(())
//! ```
//!
//! ## Prediction
//!
//! ```no_run
//! use crfs::{Attribute, Model};
//!
//! let model_data = std::fs::read("model.crfsuite")?;
//! let model = Model::new(&model_data)?;
//! let tagger = model.tagger()?;
//!
//! let xseq = vec![
//!     vec![Attribute::new("walk", 1.0)],
//!     vec![Attribute::new("shop", 1.0)],
//! ];
//! let result = tagger.tag(&xseq)?;
//! # Ok::<(), std::io::Error>(())
//! ```

mod attribute;
mod context;
mod dataset;
mod feature;
mod model;
mod tagger;

/// Training module containing all components for training CRF models
pub mod train;

// Re-export main types
pub use self::attribute::Attribute;
pub use self::model::Model;
pub use self::tagger::Tagger;

// Re-export training types for convenience
pub use self::train::Trainer;
