use std::io;

use rand::SeedableRng;
use rand::rngs::StdRng;

use super::super::crf_context::ScoreContext;
use super::super::feature_gen::FeatureGenerator;
use super::{Arow, Trainer, TrainingAlgorithm};

/// AROW training parameters.
#[derive(Debug, Clone)]
pub struct ArowParams {
    variance: f64,
    gamma: f64,
    max_iterations: usize,
    epsilon: f64,
    shuffle_seed: Option<u64>,
}

impl Default for ArowParams {
    fn default() -> Self {
        Self {
            variance: 1.0,
            gamma: 1.0,
            max_iterations: 100,
            epsilon: 1e-5,
            shuffle_seed: None,
        }
    }
}

impl ArowParams {
    pub fn variance(&self) -> f64 {
        self.variance
    }

    pub fn set_variance(&mut self, variance: f64) -> io::Result<()> {
        if variance <= 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "variance must be positive",
            ));
        }
        self.variance = variance;
        Ok(())
    }

    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    pub fn set_gamma(&mut self, gamma: f64) -> io::Result<()> {
        if gamma <= 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "gamma must be positive",
            ));
        }
        self.gamma = gamma;
        Ok(())
    }

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

impl TrainingAlgorithm for Arow {
    type Params = ArowParams;

    fn train(trainer: &mut Trainer<Self>, fgen: &mut FeatureGenerator) -> io::Result<()> {
        trainer.train_arow(fgen)
    }
}

impl Trainer<Arow> {
    /// Train using AROW algorithm
    pub(super) fn train_arow(&mut self, fgen: &mut FeatureGenerator) -> io::Result<()> {
        let num_features = fgen.num_features();
        let num_labels = self.labels.len();
        let num_instances = self.instances.len() as f64;
        let max_items = self
            .instances
            .iter()
            .map(|inst| inst.num_items as usize)
            .max()
            .unwrap_or(0);

        let variance = self.params.variance();
        let gamma = self.params.gamma();
        let max_iterations = self.params.max_iterations();
        let epsilon = self.params.epsilon();
        let verbose = self.verbose;

        // Initialize weights and covariance (diagonal)
        let mut weights = vec![0.0; num_features];
        let mut covariance = vec![variance; num_features];

        // Create CRF context for Viterbi decoding
        let mut ctx = ScoreContext::new(num_labels, max_items);
        let mut order: Vec<usize> = (0..self.instances.len()).collect();
        let mut rng = match self.params.shuffle_seed() {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        if verbose {
            println!(
                "Training with AROW (variance={}, gamma={})...",
                variance, gamma
            );
        }

        for epoch in 0..max_iterations {
            let mut sum_loss = 0.0;

            if order.len() > 1 {
                super::shuffle_indices(&mut order, &mut rng);
            }

            for &idx in &order {
                let inst = &self.instances[idx];
                let seq_len = inst.num_items as usize;

                // Predict with current weights using Viterbi
                fgen.set_weights(&weights);
                ctx.compute_scores(inst, fgen);
                let predicted = ctx.viterbi_decode(seq_len);

                // Compute loss (Hamming distance)
                let num_diff = predicted[..seq_len]
                    .iter()
                    .zip(&inst.labels[..seq_len])
                    .filter(|(p, l)| p != l)
                    .count();

                if num_diff > 0 {
                    let pred_score = ctx.sequence_score(&predicted);
                    let true_score = ctx.sequence_score(&inst.labels);
                    let cost = pred_score - true_score + num_diff as f64;

                    // Extract feature counts for true and predicted sequences
                    let true_counts = self.extract_features(inst, &inst.labels, fgen);
                    let pred_counts = self.extract_features(inst, &predicted, fgen);

                    // Compute feature difference
                    let mut diff = vec![0.0; num_features];
                    let mut frac = gamma;
                    for i in 0..num_features {
                        let delta = (true_counts[i] - pred_counts[i]) * inst.weight;
                        diff[i] = delta;
                        frac += delta * delta * covariance[i];
                    }

                    // Compute update magnitude.
                    let alpha = cost / frac;

                    // Update weights and covariance (diagonal approximation).
                    for i in 0..num_features {
                        let sigma = covariance[i];
                        let delta = diff[i];
                        weights[i] += alpha * sigma * delta;
                        covariance[i] = 1.0 / ((1.0 / sigma) + (delta * delta) / gamma);
                    }

                    sum_loss += cost * inst.weight;
                }
            }

            if verbose {
                let feature_norm: f64 = weights.iter().map(|w| w * w).sum::<f64>().sqrt();
                println!(
                    "Epoch {}: loss = {:.6}, feature_norm = {:.6}",
                    epoch + 1,
                    sum_loss,
                    feature_norm
                );
            }

            if num_instances > 0.0 && sum_loss / num_instances <= epsilon {
                if verbose {
                    println!("Converged at epoch {}", epoch + 1);
                }
                break;
            }
        }

        // Update feature weights
        fgen.set_weights(&weights);

        Ok(())
    }
}
