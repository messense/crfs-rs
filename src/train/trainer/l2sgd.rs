use std::io;

use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::StdRng};

use super::super::crf_context::ForwardBackwardContext;
use super::super::feature_gen::FeatureGenerator;
use super::{L2Sgd, Trainer, TrainingAlgorithm};

/// L2SGD training parameters.
#[derive(Debug, Clone)]
pub struct L2SgdParams {
    c2: f64,
    max_iterations: usize,
    period: usize,
    delta: f64,
    calibration_eta: f64,
    calibration_rate: f64,
    calibration_samples: usize,
    calibration_candidates: usize,
    calibration_max_trials: usize,
    shuffle_seed: Option<u64>,
}

impl Default for L2SgdParams {
    fn default() -> Self {
        Self {
            c2: 1.0,
            max_iterations: 1000,
            period: 10,
            delta: 1e-6,
            calibration_eta: 0.1,
            calibration_rate: 2.0,
            calibration_samples: 1000,
            calibration_candidates: 10,
            calibration_max_trials: 20,
            shuffle_seed: None,
        }
    }
}

impl L2SgdParams {
    pub fn c2(&self) -> f64 {
        self.c2
    }

    pub fn set_c2(&mut self, c2: f64) -> io::Result<()> {
        if c2 < 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "c2 must be non-negative",
            ));
        }
        self.c2 = c2;
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

    pub fn period(&self) -> usize {
        self.period
    }

    pub fn set_period(&mut self, period: usize) -> io::Result<()> {
        if period == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "period must be positive",
            ));
        }
        self.period = period;
        Ok(())
    }

    pub fn delta(&self) -> f64 {
        self.delta
    }

    pub fn set_delta(&mut self, delta: f64) -> io::Result<()> {
        if delta <= 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "delta must be positive",
            ));
        }
        self.delta = delta;
        Ok(())
    }

    pub fn calibration_eta(&self) -> f64 {
        self.calibration_eta
    }

    pub fn set_calibration_eta(&mut self, calibration_eta: f64) -> io::Result<()> {
        if calibration_eta <= 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "calibration.eta must be positive",
            ));
        }
        self.calibration_eta = calibration_eta;
        Ok(())
    }

    pub fn calibration_rate(&self) -> f64 {
        self.calibration_rate
    }

    pub fn set_calibration_rate(&mut self, calibration_rate: f64) -> io::Result<()> {
        if calibration_rate <= 1.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "calibration.rate must be greater than 1.0",
            ));
        }
        self.calibration_rate = calibration_rate;
        Ok(())
    }

    pub fn calibration_samples(&self) -> usize {
        self.calibration_samples
    }

    pub fn set_calibration_samples(&mut self, calibration_samples: usize) -> io::Result<()> {
        if calibration_samples == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "calibration.samples must be positive",
            ));
        }
        self.calibration_samples = calibration_samples;
        Ok(())
    }

    pub fn calibration_candidates(&self) -> usize {
        self.calibration_candidates
    }

    pub fn set_calibration_candidates(&mut self, calibration_candidates: usize) -> io::Result<()> {
        if calibration_candidates == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "calibration.candidates must be positive",
            ));
        }
        self.calibration_candidates = calibration_candidates;
        Ok(())
    }

    pub fn calibration_max_trials(&self) -> usize {
        self.calibration_max_trials
    }

    pub fn set_calibration_max_trials(&mut self, calibration_max_trials: usize) -> io::Result<()> {
        if calibration_max_trials == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "calibration.max_trials must be positive",
            ));
        }
        self.calibration_max_trials = calibration_max_trials;
        Ok(())
    }

    pub fn shuffle_seed(&self) -> Option<u64> {
        self.shuffle_seed
    }

    pub fn set_shuffle_seed(&mut self, seed: Option<u64>) {
        self.shuffle_seed = seed;
    }
}

impl TrainingAlgorithm for L2Sgd {
    type Params = L2SgdParams;

    fn train(trainer: &mut Trainer<Self>, fgen: &mut FeatureGenerator) -> io::Result<()> {
        trainer.train_l2sgd(fgen)
    }
}

impl Trainer<L2Sgd> {
    /// Train using L2SGD algorithm
    pub(super) fn train_l2sgd(&mut self, fgen: &mut FeatureGenerator) -> io::Result<()> {
        let num_features = fgen.num_features();
        let num_labels = self.labels.len();
        let max_items = self
            .instances
            .iter()
            .map(|inst| inst.num_items as usize)
            .max()
            .unwrap_or(0);

        let c2 = self.params.c2();
        let max_iterations = self.params.max_iterations();
        let period = self.params.period();
        let delta = self.params.delta();
        let verbose = self.verbose;

        let mut weights = vec![0.0; num_features];
        let num_instances = self.instances.len();
        let lambda = 2.0 * c2 / num_instances as f64;

        // Create CRF context
        let mut ctx = ForwardBackwardContext::new(num_labels, max_items);

        if verbose {
            println!("Training with L2SGD (c2={})...", c2);
        }

        let mut rng = match self.params.shuffle_seed() {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // Calibration phase: find optimal learning rate
        let t0 = self.calibrate_learning_rate(fgen, &mut ctx, lambda, &mut rng)?;

        if verbose {
            let eta = 1.0 / (lambda * t0);
            println!("Calibrated learning rate: {:.6}", eta);
        }

        // Training loop
        let mut indices: Vec<usize> = (0..self.instances.len()).collect();
        let mut objective_history = vec![0.0; period];
        let mut best_objective = f64::INFINITY;
        let mut best_weights = vec![0.0; num_features];
        let mut t = 0.0f64;

        for epoch in 1..=max_iterations {
            // Shuffle instances for better convergence
            indices.shuffle(&mut rng);

            let mut sum_loss = 0.0;
            let mut loss = 0.0;
            let mut expected = vec![0.0; num_features];
            let mut observed = vec![0.0; num_features];

            for &idx in &indices {
                let inst = &self.instances[idx];
                let seq_len = inst.num_items as usize;

                // Compute learning rate with decay
                let eta = 1.0 / (lambda * (t0 + t));

                // Apply weight decay (L2 regularization)
                let decay = 1.0 - eta * lambda;
                for w in &mut weights {
                    *w *= decay;
                }

                // Compute scores and run forward-backward
                fgen.set_weights(&weights);
                ctx.compute_scores(inst, fgen);
                let log_z = ctx.forward(seq_len);
                ctx.backward(seq_len);
                ctx.compute_marginals(seq_len, log_z);

                // Compute expected and observed counts
                expected.fill(0.0);
                observed.fill(0.0);
                ctx.expected_counts_into(inst, fgen, &mut expected);
                ctx.observed_counts_into(inst, fgen, &mut observed);

                // Update weights: w += eta * (observed - expected)
                let inst_weight = inst.weight;
                for i in 0..num_features {
                    weights[i] += eta * (observed[i] - expected[i]) * inst_weight;
                }

                // Compute loss for this instance
                loss = -ctx.log_likelihood(inst, log_z) * inst_weight;
                sum_loss += loss;
                t += 1.0;
            }

            if !loss.is_finite() {
                return Err(io::Error::other("L2SGD overflow loss"));
            }

            // Include the L2 norm of feature weights to the objective.
            let norm2: f64 = weights.iter().map(|w| w * w).sum();
            sum_loss += 0.5 * lambda * norm2 * num_instances as f64;

            if verbose {
                println!(
                    "Epoch {}: loss = {:.6}, feature_norm = {:.6}",
                    epoch,
                    sum_loss,
                    norm2.sqrt()
                );
            }

            if sum_loss < best_objective {
                best_objective = sum_loss;
                best_weights.clone_from_slice(&weights);
            }

            let improvement = if epoch > period {
                let prev = objective_history[(epoch - 1) % period];
                (prev - sum_loss) / sum_loss
            } else {
                delta
            };

            objective_history[(epoch - 1) % period] = sum_loss;

            if verbose && epoch > period {
                println!("Improvement ratio: {:.6}", improvement);
            }

            if epoch > period && improvement < delta {
                if verbose {
                    println!("Converged at epoch {}", epoch);
                }
                break;
            }
        }

        // Update feature weights
        fgen.set_weights(&best_weights);

        Ok(())
    }

    /// Calibrate learning rate for L2SGD
    fn calibrate_learning_rate(
        &self,
        fgen: &mut FeatureGenerator,
        ctx: &mut ForwardBackwardContext,
        lambda: f64,
        rng: &mut StdRng,
    ) -> io::Result<f64> {
        let num_features = fgen.num_features();
        let num_instances = self.instances.len();

        // Select calibration samples
        let num_samples = self.params.calibration_samples().min(num_instances);
        let mut sample_indices: Vec<usize> = (0..num_instances).collect();
        sample_indices.shuffle(rng);
        sample_indices.truncate(num_samples);

        let mut eta = self.params.calibration_eta();
        let mut best_eta = eta;
        let mut best_loss = f64::INFINITY;
        let mut dec = false;
        let mut num = self.params.calibration_candidates();
        let mut trials = 1;

        // Compute the initial loss without instance weights.
        let mut weights = vec![0.0; num_features];
        let mut initial_loss = 0.0;
        fgen.set_weights(&weights);
        for &idx in &sample_indices {
            let inst = &self.instances[idx];
            let seq_len = inst.num_items as usize;
            ctx.compute_scores(inst, fgen);
            let log_z = ctx.forward(seq_len);
            ctx.backward(seq_len);
            initial_loss += -ctx.log_likelihood(inst, log_z);
        }

        while num > 0 || !dec {
            let t0 = 1.0 / (lambda * eta);
            let mut t = 0.0f64;
            let mut sum_loss = 0.0;
            let mut loss = 0.0;
            let mut expected = vec![0.0; num_features];
            let mut observed = vec![0.0; num_features];
            weights.fill(0.0);

            // Perform SGD for one epoch using the calibration samples.
            for &idx in &sample_indices {
                let inst = &self.instances[idx];
                let seq_len = inst.num_items as usize;

                let eta_step = 1.0 / (lambda * (t0 + t));
                let decay = 1.0 - eta_step * lambda;
                for w in &mut weights {
                    *w *= decay;
                }

                fgen.set_weights(&weights);
                ctx.compute_scores(inst, fgen);
                let log_z = ctx.forward(seq_len);
                ctx.backward(seq_len);
                ctx.compute_marginals(seq_len, log_z);

                expected.fill(0.0);
                observed.fill(0.0);
                ctx.expected_counts_into(inst, fgen, &mut expected);
                ctx.observed_counts_into(inst, fgen, &mut observed);

                let inst_weight = inst.weight;
                for i in 0..num_features {
                    weights[i] += eta_step * (observed[i] - expected[i]) * inst_weight;
                }

                loss = -ctx.log_likelihood(inst, log_z) * inst_weight;
                sum_loss += loss;
                t += 1.0;
            }

            if !loss.is_finite() {
                sum_loss = loss;
            } else {
                let norm2: f64 = weights.iter().map(|w| w * w).sum();
                sum_loss += 0.5 * lambda * norm2 * num_samples as f64;
            }

            let ok = sum_loss.is_finite() && sum_loss < initial_loss;
            if ok {
                num = num.saturating_sub(1);
            }

            if sum_loss.is_finite() && sum_loss < best_loss {
                best_loss = sum_loss;
                best_eta = eta;
            }

            if !dec {
                if ok && num > 0 {
                    eta *= self.params.calibration_rate();
                } else {
                    dec = true;
                    num = self.params.calibration_candidates();
                    eta = self.params.calibration_eta() / self.params.calibration_rate();
                }
            } else {
                eta /= self.params.calibration_rate();
            }

            trials += 1;
            if self.params.calibration_max_trials() <= trials {
                break;
            }
        }

        Ok(1.0 / (lambda * best_eta))
    }
}
