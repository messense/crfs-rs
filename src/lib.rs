//! Pure Rust implementation of Conditional Random Fields (CRF)
//!
//! This library provides both training and prediction capabilities for linear-chain CRFs.
//!
//! # Examples
//!
//! ## Training
//!
//! ```no_run
//! use crfs::train::{Algorithm, Attribute, Trainer};
//!
//! let mut trainer = Trainer::new(true);
//! trainer.select(Algorithm::LBFGS)?;
//!
//! let xseq = vec![
//!     vec![Attribute::new("walk", 1.0)],
//!     vec![Attribute::new("shop", 1.0)],
//! ];
//! let yseq = vec!["sunny", "rainy"];
//! trainer.append(&xseq, &yseq)?;
//!
//! trainer.set("c2", "1.0")?;
//! trainer.train("model.crfsuite")?;
//! # Ok::<(), std::io::Error>(())
//! ```
//!
//! ## Prediction
//!
//! ```no_run
//! use crfs::{Model, TaggerAttribute};
//!
//! let model_data = std::fs::read("model.crfsuite")?;
//! let model = Model::new(&model_data)?;
//! let mut tagger = model.tagger()?;
//!
//! let xseq = vec![
//!     vec![TaggerAttribute::new("walk", 1.0)],
//!     vec![TaggerAttribute::new("shop", 1.0)],
//! ];
//! let result = tagger.tag(&xseq)?;
//! # Ok::<(), std::io::Error>(())
//! ```

mod context;
mod dataset;
mod feature;
mod model;
mod tagger;

/// Training module containing all components for training CRF models
pub mod train;

// Re-export main types for prediction
pub use self::model::Model;
pub use self::tagger::{Attribute as TaggerAttribute, Tagger};

// Re-export training types for convenience
pub use self::train::{Algorithm, Attribute, Trainer};
