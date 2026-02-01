//! Training module for CRF models
//!
//! This module contains all the components needed to train CRF models,
//! including feature generation, optimization, and model serialization.

mod crf_context;
mod dictionary;
mod feature_gen;
mod model_writer;
mod trainer;

// Re-export public types
pub use self::trainer::{
    Arow, ArowParams, AveragedPerceptron, AveragedPerceptronParams, L2Sgd, L2SgdParams, Lbfgs,
    LbfgsParams, LineSearchAlgorithm, PaType, PassiveAggressive, PassiveAggressiveParams, Trainer,
};
