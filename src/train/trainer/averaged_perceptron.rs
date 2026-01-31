use std::io;

use rand::SeedableRng;
use rand::rngs::StdRng;

use super::super::crf_context::ScoreContext;
use super::super::feature_gen::FeatureGenerator;
use super::{AveragedPerceptron, Trainer, TrainingAlgorithm};

/// Averaged Perceptron training parameters.
#[derive(Debug, Clone)]
pub struct AveragedPerceptronParams {
    max_iterations: usize,
    epsilon: f64,
    shuffle_seed: Option<u64>,
}

impl Default for AveragedPerceptronParams {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            epsilon: 1e-5,
            shuffle_seed: None,
        }
    }
}

impl AveragedPerceptronParams {
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    pub fn set_max_iterations(&mut self, max_iterations: usize) -> io::Result<()> {
        if max_iterations < 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "max_iterations must be at least 1",
            ));
        }
        self.max_iterations = max_iterations;
        Ok(())
    }

    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    pub fn set_epsilon(&mut self, epsilon: f64) -> io::Result<()> {
        if epsilon < 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "epsilon must be non-negative",
            ));
        }
        self.epsilon = epsilon;
        Ok(())
    }

    pub fn shuffle_seed(&self) -> Option<u64> {
        self.shuffle_seed
    }

    pub fn set_shuffle_seed(&mut self, seed: Option<u64>) {
        self.shuffle_seed = seed;
    }
}

impl TrainingAlgorithm for AveragedPerceptron {
    type Params = AveragedPerceptronParams;

    fn train(trainer: &mut Trainer<Self>, fgen: &mut FeatureGenerator) -> io::Result<()> {
        trainer.train_averaged_perceptron(fgen)
    }
}

impl Trainer<AveragedPerceptron> {
    /// Train using Averaged Perceptron algorithm
    pub(super) fn train_averaged_perceptron(
        &mut self,
        fgen: &mut FeatureGenerator,
    ) -> io::Result<()> {
        let num_features = fgen.num_features();
        let num_labels = self.labels.len();
        let num_instances = self.instances.len() as f64;
        let max_items = self
            .instances
            .iter()
            .map(|inst| inst.num_items as usize)
            .max()
            .unwrap_or(0);

        // Initialize weights and averaged weights to zero
        let mut weights = vec![0.0; num_features];
        let mut summed_updates = vec![0.0; num_features];
        let mut c = 1.0; // Update counter

        let max_iterations = self.params.max_iterations();
        let epsilon = self.params.epsilon();
        let verbose = self.verbose;

        // Create CRF context
        let mut ctx = ScoreContext::new(num_labels, max_items);
        let mut order: Vec<usize> = (0..self.instances.len()).collect();
        let mut rng = match self.params.shuffle_seed() {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        if verbose {
            println!("Training with Averaged Perceptron...");
        }

        // Training loop
        for epoch in 0..max_iterations {
            let mut loss = 0.0;

            if order.len() > 1 {
                super::shuffle_indices(&mut order, &mut rng);
            }

            for &idx in &order {
                let inst = &self.instances[idx];
                let seq_len = inst.num_items as usize;

                // Predict with current weights
                fgen.set_weights(&weights);
                ctx.compute_scores(inst, fgen);
                let predicted = ctx.viterbi_decode(seq_len);

                // Check if prediction matches true labels
                let num_diff = predicted[..seq_len]
                    .iter()
                    .zip(&inst.labels[..seq_len])
                    .filter(|(p, l)| p != l)
                    .count();

                if num_diff > 0 {
                    // Extract features for true and predicted labels
                    let true_counts = self.extract_features(inst, &inst.labels, fgen);
                    let pred_counts = self.extract_features(inst, &predicted, fgen);
                    let inst_weight = inst.weight;

                    // Update weights: w += true_features - predicted_features
                    for i in 0..num_features {
                        let delta = (true_counts[i] - pred_counts[i]) * inst_weight;
                        weights[i] += delta;
                        summed_updates[i] += c * delta;
                    }

                    // Loss is the ratio of wrongly predicted labels
                    loss += num_diff as f64 / seq_len as f64 * inst_weight;
                }

                c += 1.0;
            }

            // Check stopping criterion (error rate)
            let error_rate = if num_instances > 0.0 {
                loss / num_instances
            } else {
                0.0
            };

            if verbose {
                println!(
                    "Epoch {}: loss = {:.6} (avg per instance)",
                    epoch + 1,
                    error_rate
                );
            }

            if error_rate < epsilon {
                if verbose {
                    println!("Converged at epoch {}", epoch + 1);
                }
                break;
            }
        }

        // Average the weights
        for i in 0..num_features {
            weights[i] -= summed_updates[i] / c;
        }

        // Update feature weights
        fgen.set_weights(&weights);

        Ok(())
    }
}
