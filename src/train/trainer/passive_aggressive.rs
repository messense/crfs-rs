use std::io;

use rand::SeedableRng;
use rand::rngs::StdRng;

use super::super::crf_context::ScoreContext;
use super::super::feature_gen::FeatureGenerator;
use super::{PassiveAggressive, Trainer, TrainingAlgorithm};

/// PA variants for Passive Aggressive training.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaType {
    /// PA (no slack)
    Pa,
    /// PA-I (soft margin)
    PaI,
    /// PA-II (squared slack)
    PaII,
}

/// Passive Aggressive training parameters.
#[derive(Debug, Clone)]
pub struct PassiveAggressiveParams {
    pa_type: PaType,
    pa_c: f64,
    pa_error_sensitive: bool,
    pa_averaging: bool,
    max_iterations: usize,
    epsilon: f64,
    shuffle_seed: Option<u64>,
}

impl Default for PassiveAggressiveParams {
    fn default() -> Self {
        Self {
            pa_type: PaType::PaI,
            pa_c: 1.0,
            pa_error_sensitive: true,
            pa_averaging: true,
            max_iterations: 100,
            epsilon: 0.0,
            shuffle_seed: None,
        }
    }
}

impl PassiveAggressiveParams {
    pub fn pa_type(&self) -> PaType {
        self.pa_type
    }

    pub fn set_pa_type(&mut self, pa_type: PaType) {
        self.pa_type = pa_type;
    }

    pub fn pa_c(&self) -> f64 {
        self.pa_c
    }

    pub fn set_pa_c(&mut self, pa_c: f64) -> io::Result<()> {
        if pa_c <= 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "c must be positive",
            ));
        }
        self.pa_c = pa_c;
        Ok(())
    }

    pub fn pa_error_sensitive(&self) -> bool {
        self.pa_error_sensitive
    }

    pub fn set_pa_error_sensitive(&mut self, enabled: bool) {
        self.pa_error_sensitive = enabled;
    }

    pub fn pa_averaging(&self) -> bool {
        self.pa_averaging
    }

    pub fn set_pa_averaging(&mut self, enabled: bool) {
        self.pa_averaging = enabled;
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

impl TrainingAlgorithm for PassiveAggressive {
    type Params = PassiveAggressiveParams;

    fn train(trainer: &mut Trainer<Self>, fgen: &mut FeatureGenerator) -> io::Result<()> {
        trainer.train_passive_aggressive(fgen)
    }
}

impl Trainer<PassiveAggressive> {
    /// Train using Passive Aggressive algorithm
    pub(super) fn train_passive_aggressive(
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

        // Initialize weights to zero
        let mut weights = vec![0.0; num_features];
        let mut summed_updates = vec![0.0; num_features];
        let mut update_counter = 1.0;
        let c = self.params.pa_c();
        let pa_type = self.params.pa_type();
        let error_sensitive = self.params.pa_error_sensitive();
        let averaging = self.params.pa_averaging();
        let max_iterations = self.params.max_iterations();
        let epsilon = self.params.epsilon();
        let verbose = self.verbose;

        // Create CRF context
        let mut ctx = ScoreContext::new(num_labels, max_items);
        let mut order: Vec<usize> = (0..self.instances.len()).collect();
        let mut rng = match self.params.shuffle_seed() {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let mut thread_rng = rand::rng();
                StdRng::from_rng(&mut thread_rng)
            }
        };

        if verbose {
            println!("Training with Passive Aggressive (PA-{:?})...", pa_type);
        }

        // Training loop
        for epoch in 0..max_iterations {
            let mut sum_loss = 0.0;

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

                // Compute Hamming distance (number of incorrect labels)
                let num_diff = predicted[..seq_len]
                    .iter()
                    .zip(&inst.labels[..seq_len])
                    .filter(|(p, l)| p != l)
                    .count();

                if num_diff > 0 {
                    let pred_score = ctx.sequence_score(&predicted);
                    let true_score = ctx.sequence_score(&inst.labels);
                    let err = pred_score - true_score;
                    let cost = if error_sensitive {
                        err + (num_diff as f64).sqrt()
                    } else {
                        err + 1.0
                    };

                    // Extract features for true and predicted labels
                    let true_counts = self.extract_features(inst, &inst.labels, fgen);
                    let pred_counts = self.extract_features(inst, &predicted, fgen);

                    // Compute feature difference
                    let mut diff = vec![0.0; num_features];
                    let mut norm_sq = 0.0;
                    for i in 0..num_features {
                        let delta = true_counts[i] - pred_counts[i];
                        diff[i] = delta;
                        norm_sq += delta * delta;
                    }

                    // Compute update magnitude (tau) based on PA variant
                    let tau = if norm_sq > 0.0 {
                        match pa_type {
                            PaType::Pa => {
                                // PA (no slack): tau = cost / ||diff||^2
                                cost / norm_sq
                            }
                            PaType::PaI => {
                                // PA-I (soft margin): tau = min(C, cost / ||diff||^2)
                                (cost / norm_sq).min(c)
                            }
                            PaType::PaII => {
                                // PA-II (squared slack): tau = cost / (||diff||^2 + 1/(2*C))
                                cost / (norm_sq + 1.0 / (2.0 * c))
                            }
                        }
                    } else {
                        0.0
                    };

                    // Update weights: w += tau * diff
                    let scaled_tau = tau * inst.weight;
                    for i in 0..num_features {
                        let delta = diff[i];
                        weights[i] += scaled_tau * delta;
                        if averaging {
                            summed_updates[i] += scaled_tau * update_counter * delta;
                        }
                    }

                    sum_loss += cost * inst.weight;
                }

                update_counter += 1.0;
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

            if num_instances > 0.0 && sum_loss / num_instances < epsilon {
                if verbose {
                    println!("Converged at epoch {}", epoch + 1);
                }
                break;
            }
        }

        // Update feature weights
        if averaging {
            for i in 0..num_features {
                weights[i] -= summed_updates[i] / update_counter;
            }
        }
        fgen.set_weights(&weights);

        Ok(())
    }
}
