use std::io;

use super::crf_context::CrfContext;
use super::dictionary::Dictionary;
use super::feature_gen::FeatureGenerator;
use super::model_writer::ModelWriter;
use crate::attribute::Attribute;
use crate::dataset::{Attribute as DatasetAttribute, Instance};

/// Training algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// L-BFGS optimization
    LBFGS,
}

/// Training parameters
#[derive(Debug, Clone)]
pub struct TrainingParams {
    /// L1 regularization coefficient
    pub c1: f64,
    /// L2 regularization coefficient
    pub c2: f64,
    /// Number of limited memories for L-BFGS
    pub num_memories: usize,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence epsilon
    pub epsilon: f64,
    /// Minimum feature frequency
    pub feature_minfreq: f64,
    /// Enable verbose output
    pub verbose: bool,
}

impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            c1: 0.0,
            c2: 1.0,
            num_memories: 6,
            max_iterations: 100,
            epsilon: 1e-5,
            feature_minfreq: 0.0,
            verbose: false,
        }
    }
}

/// CRF Trainer
#[derive(Debug)]
pub struct Trainer {
    /// Training instances
    instances: Vec<Instance>,
    /// Attribute dictionary
    attrs: Dictionary,
    /// Label dictionary
    labels: Dictionary,
    /// Selected algorithm
    algorithm: Option<Algorithm>,
    /// Training parameters
    params: TrainingParams,
}

impl Trainer {
    /// Create a new trainer
    pub fn new(verbose: bool) -> Self {
        Self {
            instances: Vec::new(),
            attrs: Dictionary::new(),
            labels: Dictionary::new(),
            algorithm: None,
            params: TrainingParams {
                verbose,
                ..Default::default()
            },
        }
    }

    /// Select training algorithm
    pub fn select(&mut self, algorithm: Algorithm) -> io::Result<()> {
        self.algorithm = Some(algorithm);
        Ok(())
    }

    /// Append training data
    pub fn append<I, L>(&mut self, xseq: &[I], yseq: &[L]) -> io::Result<()>
    where
        I: AsRef<[Attribute]>,
        L: AsRef<str>,
    {
        if xseq.len() != yseq.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "xseq and yseq must have the same length",
            ));
        }

        let mut instance = Instance::with_capacity(xseq.len());

        for (item, label) in xseq.iter().zip(yseq.iter()) {
            // Convert attributes to dataset format with IDs
            let mut dataset_item = Vec::new();
            for attr in item.as_ref() {
                let aid = self.attrs.get_or_insert(&attr.name);
                dataset_item.push(DatasetAttribute::new(aid, attr.value));
            }

            // Get or create label ID
            let lid = self.labels.get_or_insert(label.as_ref());

            instance.push(dataset_item, lid);
        }

        self.instances.push(instance);
        Ok(())
    }

    /// Clear all training data
    pub fn clear(&mut self) {
        self.instances.clear();
        self.attrs.clear();
        self.labels.clear();
    }

    /// Set a training parameter
    pub fn set(&mut self, name: &str, value: &str) -> io::Result<()> {
        match name {
            "c1" => {
                let c1 = value
                    .parse()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid c1 value"))?;
                if c1 < 0.0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "c1 must be non-negative",
                    ));
                }
                self.params.c1 = c1;
            }
            "c2" => {
                let c2 = value
                    .parse()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid c2 value"))?;
                if c2 < 0.0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "c2 must be non-negative",
                    ));
                }
                self.params.c2 = c2;
            }
            "num_memories" => {
                let num = value.parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "invalid num_memories value")
                })?;
                if num < 1 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "num_memories must be at least 1",
                    ));
                }
                self.params.num_memories = num;
            }
            "max_iterations" => {
                let max_iterations = value.parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "invalid max_iterations value")
                })?;
                if max_iterations < 1 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "max_iterations must be at least 1",
                    ));
                }
                self.params.max_iterations = max_iterations;
            }
            "epsilon" => {
                let epsilon = value.parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "invalid epsilon value")
                })?;
                if epsilon <= 0.0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "epsilon must be > 0.0",
                    ));
                }
                self.params.epsilon = epsilon;
            }
            "feature.minfreq" => {
                let feature_minfreq = value.parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "invalid feature.minfreq value")
                })?;
                if feature_minfreq < 0.0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "feature.minfreq must be non-negative",
                    ));
                }
                self.params.feature_minfreq = feature_minfreq;
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown parameter: {}", name),
                ));
            }
        }
        Ok(())
    }

    /// Get a training parameter
    pub fn get(&self, name: &str) -> io::Result<String> {
        match name {
            "c1" => Ok(self.params.c1.to_string()),
            "c2" => Ok(self.params.c2.to_string()),
            "num_memories" => Ok(self.params.num_memories.to_string()),
            "max_iterations" => Ok(self.params.max_iterations.to_string()),
            "epsilon" => Ok(self.params.epsilon.to_string()),
            "feature.minfreq" => Ok(self.params.feature_minfreq.to_string()),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unknown parameter: {}", name),
            )),
        }
    }

    /// Train the model and save to file
    pub fn train(&mut self, filename: &str) -> io::Result<()> {
        if self.algorithm.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "algorithm not selected",
            ));
        }

        if self.instances.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "no training data",
            ));
        }

        // Generate features
        if self.params.verbose {
            println!("Generating features...");
        }
        let mut fgen = FeatureGenerator::generate(
            &self.instances,
            &self.attrs,
            &self.labels,
            self.params.feature_minfreq,
        )?;

        if self.params.verbose {
            println!("Number of features: {}", fgen.num_features());
            println!("Number of labels: {}", self.labels.len());
            println!("Number of attributes: {}", self.attrs.len());
        }

        // Train with LBFGS
        match self.algorithm {
            Some(Algorithm::LBFGS) => self.train_lbfgs(&mut fgen)?,
            None => unreachable!(),
        }

        // Save model
        if self.params.verbose {
            println!("Saving model to {}...", filename);
        }
        ModelWriter::write(filename, &fgen, &self.labels, &self.attrs)?;

        if self.params.verbose {
            println!("Training completed.");
        }

        Ok(())
    }

    /// Train using L-BFGS algorithm
    fn train_lbfgs(&mut self, fgen: &mut FeatureGenerator) -> io::Result<()> {
        let num_features = fgen.num_features();
        let num_labels = self.labels.len();
        let max_items = self
            .instances
            .iter()
            .map(|inst| inst.num_items as usize)
            .max()
            .unwrap_or(0);

        // Initialize weights to zero
        let mut weights = vec![0.0; num_features];

        // Create CRF context
        let mut ctx = CrfContext::new(num_labels, max_items);

        // Pre-allocate vectors to avoid repeated allocations in the optimization loop
        let mut gradient = vec![0.0; num_features];
        let mut expected = vec![0.0; num_features];
        let mut observed = vec![0.0; num_features];

        // Objective function: negative log-likelihood + L2 regularization
        let evaluate = |x: &[f64], gx: &mut [f64]| -> Result<f64, anyhow::Error> {
            // Update feature weights
            fgen.set_weights(x);

            let mut loss = 0.0;
            gradient.fill(0.0);

            // Compute loss and gradient for each instance
            for inst in &self.instances {
                let seq_len = inst.num_items as usize;

                // Compute scores
                // Note: log_likelihood() below will recompute scores internally.
                // This is redundant but necessary because forward() needs scores.
                // A future optimization could split log_likelihood to avoid recomputation.
                ctx.compute_scores(inst, fgen);

                // Forward-backward
                let log_z = ctx.forward(seq_len);
                ctx.backward(seq_len);
                ctx.compute_marginals(seq_len, log_z);

                // Log-likelihood (recomputes scores internally)
                let log_likelihood = ctx.log_likelihood(inst, fgen);
                loss -= log_likelihood;

                // Gradient = expected - observed
                // Reuse pre-allocated vectors
                expected.fill(0.0);
                observed.fill(0.0);
                ctx.expected_counts_into(inst, fgen, &mut expected);
                ctx.observed_counts_into(inst, fgen, &mut observed);
                for i in 0..num_features {
                    gradient[i] += expected[i] - observed[i];
                }
            }

            // Add L2 regularization
            // Factor of 2 comes from derivative of c2 * x[i]^2 -> 2 * c2 * x[i]
            if self.params.c2 > 0.0 {
                let two_c2 = self.params.c2 * 2.0;
                for i in 0..num_features {
                    gradient[i] += two_c2 * x[i];
                    loss += self.params.c2 * x[i] * x[i];
                }
            }

            // Copy gradient
            gx.copy_from_slice(&gradient);

            Ok(loss)
        };

        // Progress callback
        let progress = |prgr: &liblbfgs::Progress| -> bool {
            if self.params.verbose {
                println!(
                    "Iteration {}: loss = {:.6}, ||x|| = {:.6}, ||g|| = {:.6}",
                    prgr.niter, prgr.fx, prgr.xnorm, prgr.gnorm
                );
            }
            false // continue optimization
        };

        // Run L-BFGS optimization
        // Note: num_memories is stored but liblbfgs doesn't expose a way to set it
        let mut lbfgs = liblbfgs::lbfgs()
            .with_max_iterations(self.params.max_iterations)
            .with_epsilon(self.params.epsilon);

        // Add L1 regularization if c1 > 0
        if self.params.c1 > 0.0 {
            lbfgs = lbfgs.with_orthantwise(self.params.c1, 0, num_features);
        }

        let result = lbfgs
            .minimize(&mut weights, evaluate, progress)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("LBFGS error: {}", e)))?;

        if self.params.verbose {
            println!("Final loss: {:.6}", result.fx);
        }

        // Update feature weights
        fgen.set_weights(&weights);

        Ok(())
    }
}

impl Default for Trainer {
    fn default() -> Self {
        Self::new(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attribute_creation() {
        let attr1 = Attribute::new("walk", 1.0);
        assert_eq!(attr1.name, "walk");
        assert_eq!(attr1.value, 1.0);

        let attr2 = Attribute::from("shop");
        assert_eq!(attr2.name, "shop");
        assert_eq!(attr2.value, 1.0);

        let attr3 = Attribute::from(("clean", 0.5));
        assert_eq!(attr3.name, "clean");
        assert_eq!(attr3.value, 0.5);
    }

    #[test]
    fn test_trainer_basic() {
        let mut trainer = Trainer::new(false);
        assert!(trainer.select(Algorithm::LBFGS).is_ok());

        let xseq = vec![
            vec![Attribute::new("walk", 1.0), Attribute::new("shop", 0.5)],
            vec![Attribute::new("walk", 1.0)],
        ];
        let yseq = vec!["sunny", "sunny"];

        assert!(trainer.append(&xseq, &yseq).is_ok());
        assert_eq!(trainer.instances.len(), 1);
        assert_eq!(trainer.attrs.len(), 2); // walk, shop
        assert_eq!(trainer.labels.len(), 1); // sunny
    }

    #[test]
    fn test_trainer_params() {
        let mut trainer = Trainer::new(false);
        assert!(trainer.set("c1", "0.5").is_ok());
        assert!(trainer.set("c2", "2.0").is_ok());
        assert_eq!(trainer.get("c1").unwrap(), "0.5");
        assert_eq!(trainer.get("c2").unwrap(), "2");

        assert!(trainer.set("invalid_param", "1.0").is_err());
        assert!(trainer.get("invalid_param").is_err());
    }
}
