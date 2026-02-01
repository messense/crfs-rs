use std::io;

use super::super::crf_context::ForwardBackwardContext;
use super::super::feature_gen::FeatureGenerator;
use super::{Lbfgs, Trainer, TrainingAlgorithm};

/// Line search algorithm for L-BFGS optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LineSearchAlgorithm {
    /// More-Thuente line search (default, CRFsuite default)
    #[default]
    MoreThuente,
    /// Backtracking with Armijo condition
    BacktrackingArmijo,
    /// Backtracking with Wolfe condition
    BacktrackingWolfe,
    /// Backtracking with strong Wolfe condition
    BacktrackingStrongWolfe,
}

impl LineSearchAlgorithm {
    fn to_liblbfgs_str(self) -> &'static str {
        match self {
            Self::MoreThuente => "MoreThuente",
            Self::BacktrackingArmijo => "BacktrackingArmijo",
            Self::BacktrackingWolfe => "BacktrackingWolfe",
            Self::BacktrackingStrongWolfe => "BacktrackingStrongWolfe",
        }
    }
}

/// L-BFGS training parameters.
#[derive(Debug, Clone)]
pub struct LbfgsParams {
    c1: f64,
    c2: f64,
    num_memories: usize,
    max_iterations: usize,
    epsilon: f64,
    period: usize,
    delta: f64,
    linesearch: LineSearchAlgorithm,
    max_linesearch: usize,
}

impl Default for LbfgsParams {
    fn default() -> Self {
        Self {
            c1: 0.0,
            c2: 1.0,
            num_memories: 6,
            max_iterations: usize::MAX,
            epsilon: 1e-5,
            period: 10,
            delta: 1e-5,
            linesearch: LineSearchAlgorithm::default(),
            max_linesearch: 20,
        }
    }
}

impl LbfgsParams {
    pub fn c1(&self) -> f64 {
        self.c1
    }

    pub fn set_c1(&mut self, c1: f64) -> io::Result<()> {
        if c1 < 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "c1 must be non-negative",
            ));
        }
        self.c1 = c1;
        Ok(())
    }

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

    pub fn num_memories(&self) -> usize {
        self.num_memories
    }

    pub fn set_num_memories(&mut self, num_memories: usize) -> io::Result<()> {
        if num_memories < 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "num_memories must be at least 1",
            ));
        }
        self.num_memories = num_memories;
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

    pub fn period(&self) -> usize {
        self.period
    }

    /// Set the period for delta-based convergence test.
    ///
    /// Setting period to 0 disables the delta-based convergence test
    /// (only gradient-based epsilon test is used).
    pub fn set_period(&mut self, period: usize) {
        self.period = period;
    }

    pub fn delta(&self) -> f64 {
        self.delta
    }

    pub fn set_delta(&mut self, delta: f64) -> io::Result<()> {
        if delta < 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "delta must be non-negative",
            ));
        }
        self.delta = delta;
        Ok(())
    }

    pub fn linesearch(&self) -> LineSearchAlgorithm {
        self.linesearch
    }

    pub fn set_linesearch(&mut self, linesearch: LineSearchAlgorithm) {
        self.linesearch = linesearch;
    }

    pub fn max_linesearch(&self) -> usize {
        self.max_linesearch
    }

    pub fn set_max_linesearch(&mut self, max_linesearch: usize) -> io::Result<()> {
        if max_linesearch == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "max_linesearch must be positive",
            ));
        }
        self.max_linesearch = max_linesearch;
        Ok(())
    }
}

impl TrainingAlgorithm for Lbfgs {
    type Params = LbfgsParams;

    fn train(trainer: &mut Trainer<Self>, fgen: &mut FeatureGenerator) -> io::Result<()> {
        trainer.train_lbfgs(fgen)
    }
}

impl Trainer<Lbfgs> {
    /// Train using L-BFGS algorithm
    pub(super) fn train_lbfgs(&mut self, fgen: &mut FeatureGenerator) -> io::Result<()> {
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
        let mut ctx = ForwardBackwardContext::new(num_labels, max_items);

        // Pre-allocate vectors to avoid repeated allocations in the optimization loop
        let mut gradient = vec![0.0; num_features];
        let mut expected = vec![0.0; num_features];
        let mut observed = vec![0.0; num_features];

        let c1 = self.params.c1();
        let c2 = self.params.c2();
        let max_iterations = self.params.max_iterations();
        let epsilon = self.params.epsilon();
        let period = self.params.period();
        let delta = self.params.delta();
        let linesearch = self.params.linesearch();
        let max_linesearch = self.params.max_linesearch();
        let verbose = self.verbose;

        // Objective function: negative log-likelihood + L2 regularization
        let evaluate = |x: &[f64], gx: &mut [f64]| -> Result<f64, anyhow::Error> {
            // Update feature weights
            fgen.set_weights(x);

            let mut loss = 0.0;
            gradient.fill(0.0);

            // Compute loss and gradient for each instance
            for inst in &self.instances {
                let seq_len = inst.num_items as usize;
                let inst_weight = inst.weight;

                // Compute scores and run forward-backward algorithm
                ctx.compute_scores(inst, fgen);
                let log_z = ctx.forward(seq_len);
                ctx.backward(seq_len);
                ctx.compute_marginals(seq_len, log_z);

                // Compute log-likelihood using pre-computed scores and partition function
                let log_likelihood = ctx.log_likelihood(inst, log_z);
                loss -= log_likelihood * inst_weight;

                // Gradient = expected - observed, weighted by instance weight
                // Reuse pre-allocated vectors
                expected.fill(0.0);
                observed.fill(0.0);
                ctx.expected_counts_into(inst, fgen, &mut expected);
                ctx.observed_counts_into(inst, fgen, &mut observed);
                for i in 0..num_features {
                    gradient[i] += (expected[i] - observed[i]) * inst_weight;
                }
            }

            // Add L2 regularization
            // Factor of 2 comes from derivative of c2 * x[i]^2 -> 2 * c2 * x[i]
            if c2 > 0.0 {
                let two_c2 = c2 * 2.0;
                for i in 0..num_features {
                    gradient[i] += two_c2 * x[i];
                    loss += c2 * x[i] * x[i];
                }
            }

            // Copy gradient
            gx.copy_from_slice(&gradient);

            Ok(loss)
        };

        // Progress callback
        let progress = |prgr: &liblbfgs::Progress| -> bool {
            if verbose {
                println!(
                    "Iteration {}: loss = {:.6}, ||x|| = {:.6}, ||g|| = {:.6}",
                    prgr.niter, prgr.fx, prgr.xnorm, prgr.gnorm
                );
            }
            false // continue optimization
        };

        // Run L-BFGS optimization
        // Note: LbfgsParams::num_memories is accepted and stored for API compatibility,
        // but is currently ignored when configuring the LBFGS optimizer because the
        // liblbfgs crate does not expose a way to configure the number of limited
        // memory vectors used by the L-BFGS algorithm. The library uses its default.
        let mut lbfgs = liblbfgs::lbfgs()
            .with_max_iterations(max_iterations)
            .with_epsilon(epsilon)
            .with_fx_delta(delta, period)
            .with_max_linesearch(max_linesearch);

        // Add L1 regularization if c1 > 0 (OWL-QN)
        // OWL-QN only supports backtracking line search, so we force it here
        // regardless of the configured linesearch algorithm (matching CRFsuite behavior).
        if c1 > 0.0 {
            lbfgs = lbfgs
                .with_linesearch_algorithm("BacktrackingStrongWolfe")
                .with_orthantwise(c1, 0, num_features);
        } else {
            lbfgs = lbfgs.with_linesearch_algorithm(linesearch.to_liblbfgs_str());
        }

        let result = lbfgs
            .minimize(&mut weights, evaluate, progress)
            .map_err(|e| io::Error::other(format!("LBFGS error: {}", e)))?;

        if verbose {
            println!("Final loss: {:.6}", result.fx);
        }

        // Update feature weights
        fgen.set_weights(&weights);

        Ok(())
    }
}
