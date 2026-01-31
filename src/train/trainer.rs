use std::io;
use std::path::Path;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::crf_context::CrfContext;
use super::dictionary::Dictionary;
use super::feature_gen::FeatureGenerator;
use super::model_writer::ModelWriter;
use crate::attribute::Attribute;
use crate::dataset::{Attribute as DatasetAttribute, Instance};

fn shuffle_indices(indices: &mut [usize], rng: &mut StdRng) {
    let len = indices.len();
    for i in 0..len {
        let j = rng.gen_range(0..len);
        indices.swap(i, j);
    }
}

/// Training algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// L-BFGS optimization
    LBFGS,
    /// Averaged Perceptron
    AveragedPerceptron,
    /// Passive Aggressive
    PassiveAggressive,
    /// L2-regularized SGD
    L2SGD,
    /// Adaptive Regularization (AROW)
    AROW,
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
    /// Seed for shuffling training instances (None uses entropy)
    pub shuffle_seed: Option<u64>,
    /// Enable verbose output
    pub verbose: bool,
    /// PA type: 0 (PA), 1 (PA-I), 2 (PA-II)
    pub pa_type: usize,
    /// PA aggressiveness parameter
    pub pa_c: f64,
    /// PA cost function uses error-sensitive variant when true
    pub pa_error_sensitive: bool,
    /// PA weight averaging (similarly to Averaged Perceptron)
    pub pa_averaging: bool,
    /// L2SGD: Period for convergence check
    pub period: usize,
    /// L2SGD: Delta for convergence threshold
    pub delta: f64,
    /// L2SGD: Initial learning rate for calibration
    pub calibration_eta: f64,
    /// L2SGD: Rate of increase/decrease for calibration
    pub calibration_rate: f64,
    /// L2SGD: Number of samples for calibration
    pub calibration_samples: usize,
    /// L2SGD: Number of candidates for calibration
    pub calibration_candidates: usize,
    /// L2SGD: Maximum trials for calibration
    pub calibration_max_trials: usize,
    /// AROW: Initial variance for covariance matrix
    pub variance: f64,
    /// AROW: Trade-off parameter (gamma)
    pub gamma: f64,
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
            shuffle_seed: None,
            verbose: false,
            pa_type: 1,
            pa_c: 1.0,
            pa_error_sensitive: true,
            pa_averaging: true,
            period: 10,
            delta: 1e-5,
            calibration_eta: 0.1,
            calibration_rate: 2.0,
            calibration_samples: 1000,
            calibration_candidates: 10,
            calibration_max_trials: 20,
            variance: 1.0,
            gamma: 1.0,
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
    algorithm: Algorithm,
    /// Training parameters
    params: TrainingParams,
}

impl Trainer {
    /// Create a new trainer
    pub fn new(algorithm: Algorithm) -> Self {
        Self {
            instances: Vec::new(),
            attrs: Dictionary::new(),
            labels: Dictionary::new(),
            algorithm,
            params: TrainingParams::default(),
        }
    }

    /// Enable or disable verbose output
    pub fn verbose(&mut self, enabled: bool) -> &mut Self {
        self.params.verbose = enabled;
        self
    }

    /// Append training data
    pub fn append<I, L>(&mut self, xseq: &[I], yseq: &[L]) -> io::Result<()>
    where
        I: AsRef<[Attribute]>,
        L: AsRef<str>,
    {
        self.append_with_weight(xseq, yseq, 1.0)
    }

    /// Append weighted training data
    pub fn append_with_weight<I, L>(
        &mut self,
        xseq: &[I],
        yseq: &[L],
        weight: f64,
    ) -> io::Result<()>
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

        if xseq.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "empty sequences are not allowed",
            ));
        }

        let mut instance = Instance::with_capacity(xseq.len());
        instance.set_weight(weight);

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
                if epsilon < 0.0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "epsilon must be non-negative",
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
            "seed" | "shuffle.seed" => {
                if value.eq_ignore_ascii_case("auto") || value.eq_ignore_ascii_case("none") {
                    self.params.shuffle_seed = None;
                } else {
                    let seed = value.parse().map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidInput, "invalid seed value")
                    })?;
                    self.params.shuffle_seed = Some(seed);
                }
            }
            "type" => {
                let pa_type = value.parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "invalid type value")
                })?;
                if pa_type > 2 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "type must be 0, 1, or 2",
                    ));
                }
                self.params.pa_type = pa_type;
            }
            "c" => {
                let pa_c = value
                    .parse()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid c value"))?;
                if pa_c <= 0.0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "c must be positive",
                    ));
                }
                self.params.pa_c = pa_c;
            }
            "error_sensitive" => {
                let error_sensitive: u8 = value.parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "invalid error_sensitive value")
                })?;
                if error_sensitive > 1 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "error_sensitive must be 0 or 1",
                    ));
                }
                self.params.pa_error_sensitive = error_sensitive == 1;
            }
            "averaging" => {
                let averaging: u8 = value.parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "invalid averaging value")
                })?;
                if averaging > 1 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "averaging must be 0 or 1",
                    ));
                }
                self.params.pa_averaging = averaging == 1;
            }
            "period" => {
                let period = value.parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "invalid period value")
                })?;
                if period == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "period must be positive",
                    ));
                }
                self.params.period = period;
            }
            "delta" => {
                let delta = value.parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "invalid delta value")
                })?;
                if delta <= 0.0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "delta must be positive",
                    ));
                }
                self.params.delta = delta;
            }
            "calibration.eta" => {
                let eta = value.parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "invalid calibration.eta value")
                })?;
                if eta <= 0.0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "calibration.eta must be positive",
                    ));
                }
                self.params.calibration_eta = eta;
            }
            "calibration.rate" => {
                let rate = value.parse().map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "invalid calibration.rate value",
                    )
                })?;
                if rate <= 1.0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "calibration.rate must be greater than 1.0",
                    ));
                }
                self.params.calibration_rate = rate;
            }
            "calibration.samples" => {
                let samples = value.parse().map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "invalid calibration.samples value",
                    )
                })?;
                if samples == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "calibration.samples must be positive",
                    ));
                }
                self.params.calibration_samples = samples;
            }
            "calibration.candidates" => {
                let candidates = value.parse().map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "invalid calibration.candidates value",
                    )
                })?;
                if candidates == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "calibration.candidates must be positive",
                    ));
                }
                self.params.calibration_candidates = candidates;
            }
            "calibration.max_trials" => {
                let max_trials = value.parse().map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "invalid calibration.max_trials value",
                    )
                })?;
                if max_trials == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "calibration.max_trials must be positive",
                    ));
                }
                self.params.calibration_max_trials = max_trials;
            }
            "variance" => {
                let variance = value.parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "invalid variance value")
                })?;
                if variance <= 0.0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "variance must be positive",
                    ));
                }
                self.params.variance = variance;
            }
            "gamma" => {
                let gamma = value.parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "invalid gamma value")
                })?;
                if gamma <= 0.0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "gamma must be positive",
                    ));
                }
                self.params.gamma = gamma;
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
            "seed" | "shuffle.seed" => Ok(self
                .params
                .shuffle_seed
                .map(|seed| seed.to_string())
                .unwrap_or_else(|| "auto".to_string())),
            "type" => Ok(self.params.pa_type.to_string()),
            "c" => Ok(self.params.pa_c.to_string()),
            "error_sensitive" => Ok(if self.params.pa_error_sensitive {
                "1".to_string()
            } else {
                "0".to_string()
            }),
            "averaging" => Ok(if self.params.pa_averaging {
                "1".to_string()
            } else {
                "0".to_string()
            }),
            "period" => Ok(self.params.period.to_string()),
            "delta" => Ok(self.params.delta.to_string()),
            "calibration.eta" => Ok(self.params.calibration_eta.to_string()),
            "calibration.rate" => Ok(self.params.calibration_rate.to_string()),
            "calibration.samples" => Ok(self.params.calibration_samples.to_string()),
            "calibration.candidates" => Ok(self.params.calibration_candidates.to_string()),
            "calibration.max_trials" => Ok(self.params.calibration_max_trials.to_string()),
            "variance" => Ok(self.params.variance.to_string()),
            "gamma" => Ok(self.params.gamma.to_string()),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unknown parameter: {}", name),
            )),
        }
    }

    /// Train the model and save to file
    pub fn train(&mut self, filename: &Path) -> io::Result<()> {
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

        // Train with selected algorithm
        match self.algorithm {
            Algorithm::LBFGS => self.train_lbfgs(&mut fgen)?,
            Algorithm::AveragedPerceptron => self.train_averaged_perceptron(&mut fgen)?,
            Algorithm::PassiveAggressive => self.train_passive_aggressive(&mut fgen)?,
            Algorithm::L2SGD => self.train_l2sgd(&mut fgen)?,
            Algorithm::AROW => self.train_arow(&mut fgen)?,
        }

        // Save model
        if self.params.verbose {
            println!("Saving model to {}...", filename.display());
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

                // Compute scores and run forward-backward algorithm
                ctx.compute_scores(inst, fgen);
                let log_z = ctx.forward(seq_len);
                ctx.backward(seq_len);
                ctx.compute_marginals(seq_len, log_z);

                // Compute log-likelihood using pre-computed scores and partition function
                let log_likelihood = ctx.log_likelihood(inst, log_z);
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
        // Note: TrainingParams::num_memories is accepted and stored for API compatibility,
        // but is currently ignored when configuring the LBFGS optimizer because the
        // liblbfgs crate does not expose a way to configure the number of limited
        // memory vectors used by the L-BFGS algorithm. The library uses its default.
        let mut lbfgs = liblbfgs::lbfgs()
            .with_max_iterations(self.params.max_iterations)
            .with_epsilon(self.params.epsilon);

        // Add L1 regularization if c1 > 0
        if self.params.c1 > 0.0 {
            lbfgs = lbfgs.with_orthantwise(self.params.c1, 0, num_features);
        }

        let result = lbfgs
            .minimize(&mut weights, evaluate, progress)
            .map_err(|e| io::Error::other(format!("LBFGS error: {}", e)))?;

        if self.params.verbose {
            println!("Final loss: {:.6}", result.fx);
        }

        // Update feature weights
        fgen.set_weights(&weights);

        Ok(())
    }

    /// Train using Averaged Perceptron algorithm
    fn train_averaged_perceptron(&mut self, fgen: &mut FeatureGenerator) -> io::Result<()> {
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

        // Create CRF context
        let mut ctx = CrfContext::new(num_labels, max_items);
        let mut order: Vec<usize> = (0..self.instances.len()).collect();
        let mut rng = match self.params.shuffle_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        if self.params.verbose {
            println!("Training with Averaged Perceptron...");
        }

        // Training loop
        for epoch in 0..self.params.max_iterations {
            let mut loss = 0.0;

            if order.len() > 1 {
                shuffle_indices(&mut order, &mut rng);
            }

            for &idx in &order {
                let inst = &self.instances[idx];
                let seq_len = inst.num_items as usize;

                // Predict with current weights
                fgen.set_weights(&weights);
                ctx.compute_scores(inst, fgen);
                let predicted = ctx.viterbi_decode(seq_len);

                // Check if prediction matches true labels
                let mut num_diff = 0;
                for t in 0..seq_len {
                    if predicted[t] != inst.labels[t] {
                        num_diff += 1;
                    }
                }

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

            if self.params.verbose {
                println!(
                    "Epoch {}: loss = {:.6} (avg per instance)",
                    epoch + 1,
                    error_rate
                );
            }

            if error_rate < self.params.epsilon {
                if self.params.verbose {
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

    /// Extract feature counts for a given label sequence
    fn extract_features(
        &self,
        inst: &Instance,
        labels: &[u32],
        fgen: &FeatureGenerator,
    ) -> Vec<f64> {
        use super::feature_gen::FeatureType;

        let mut counts = vec![0.0; fgen.num_features()];
        let seq_len = inst.num_items as usize;

        // State features
        for t in 0..seq_len {
            let label = labels[t];
            for attr in &inst.items[t] {
                let aid = attr.id as usize;
                if aid < fgen.attr_refs.len() {
                    for &fid in &fgen.attr_refs[aid].fids {
                        let feature = &fgen.features[fid as usize];
                        if feature.ftype == FeatureType::State && feature.dst == label {
                            counts[fid as usize] += attr.value;
                        }
                    }
                }
            }
        }

        // Transition features
        for t in 1..seq_len {
            let prev_label = labels[t - 1];
            let label = labels[t];
            let prev_l = prev_label as usize;
            if prev_l < fgen.label_refs.len() {
                for &fid in &fgen.label_refs[prev_l].fids {
                    let feature = &fgen.features[fid as usize];
                    if feature.ftype == FeatureType::Transition
                        && feature.src == prev_label
                        && feature.dst == label
                    {
                        counts[fid as usize] += 1.0;
                    }
                }
            }
        }

        counts
    }

    /// Train using Passive Aggressive algorithm
    fn train_passive_aggressive(&mut self, fgen: &mut FeatureGenerator) -> io::Result<()> {
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
        let c = self.params.pa_c;
        let pa_type = self.params.pa_type;
        let error_sensitive = self.params.pa_error_sensitive;
        let averaging = self.params.pa_averaging;

        // Create CRF context
        let mut ctx = CrfContext::new(num_labels, max_items);
        let mut order: Vec<usize> = (0..self.instances.len()).collect();
        let mut rng = match self.params.shuffle_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        if self.params.verbose {
            println!("Training with Passive Aggressive (PA-{})...", pa_type);
        }

        // Training loop
        for epoch in 0..self.params.max_iterations {
            let mut sum_loss = 0.0;

            if order.len() > 1 {
                shuffle_indices(&mut order, &mut rng);
            }

            for &idx in &order {
                let inst = &self.instances[idx];
                let seq_len = inst.num_items as usize;

                // Predict with current weights
                fgen.set_weights(&weights);
                ctx.compute_scores(inst, fgen);
                let predicted = ctx.viterbi_decode(seq_len);

                // Compute Hamming distance (number of incorrect labels)
                let mut num_diff = 0;
                for t in 0..seq_len {
                    if predicted[t] != inst.labels[t] {
                        num_diff += 1;
                    }
                }

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
                            0 => {
                                // PA (no slack): tau = cost / ||diff||^2
                                cost / norm_sq
                            }
                            1 => {
                                // PA-I (soft margin): tau = min(C, cost / ||diff||^2)
                                (cost / norm_sq).min(c)
                            }
                            2 => {
                                // PA-II (squared slack): tau = cost / (||diff||^2 + 1/(2*C))
                                cost / (norm_sq + 1.0 / (2.0 * c))
                            }
                            _ => unreachable!(),
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

            if self.params.verbose {
                let feature_norm: f64 = weights.iter().map(|w| w * w).sum::<f64>().sqrt();
                println!(
                    "Epoch {}: loss = {:.6}, feature_norm = {:.6}",
                    epoch + 1,
                    sum_loss,
                    feature_norm
                );
            }

            if num_instances > 0.0 && sum_loss / num_instances < self.params.epsilon {
                if self.params.verbose {
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

    /// Train using AROW algorithm
    fn train_arow(&mut self, fgen: &mut FeatureGenerator) -> io::Result<()> {
        let num_features = fgen.num_features();
        let num_labels = self.labels.len();
        let num_instances = self.instances.len() as f64;
        let max_items = self
            .instances
            .iter()
            .map(|inst| inst.num_items as usize)
            .max()
            .unwrap_or(0);

        // Initialize weights and covariance (diagonal)
        let mut weights = vec![0.0; num_features];
        let mut covariance = vec![self.params.variance; num_features];
        let gamma = self.params.gamma;

        // Create CRF context for Viterbi decoding
        let mut ctx = CrfContext::new(num_labels, max_items);
        let mut order: Vec<usize> = (0..self.instances.len()).collect();
        let mut rng = match self.params.shuffle_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        if self.params.verbose {
            println!(
                "Training with AROW (variance={}, gamma={})...",
                self.params.variance, gamma
            );
        }

        for epoch in 0..self.params.max_iterations {
            let mut sum_loss = 0.0;

            if order.len() > 1 {
                shuffle_indices(&mut order, &mut rng);
            }

            for &idx in &order {
                let inst = &self.instances[idx];
                let seq_len = inst.num_items as usize;

                // Predict with current weights using Viterbi
                fgen.set_weights(&weights);
                ctx.compute_scores(inst, fgen);
                let predicted = ctx.viterbi_decode(seq_len);

                // Compute loss (Hamming distance)
                let mut num_diff = 0;
                for t in 0..seq_len {
                    if predicted[t] != inst.labels[t] {
                        num_diff += 1;
                    }
                }

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

            if self.params.verbose {
                let feature_norm: f64 = weights.iter().map(|w| w * w).sum::<f64>().sqrt();
                println!(
                    "Epoch {}: loss = {:.6}, feature_norm = {:.6}",
                    epoch + 1,
                    sum_loss,
                    feature_norm
                );
            }

            if num_instances > 0.0 && sum_loss / num_instances <= self.params.epsilon {
                if self.params.verbose {
                    println!("Converged at epoch {}", epoch + 1);
                }
                break;
            }
        }

        // Update feature weights
        fgen.set_weights(&weights);

        Ok(())
    }

    /// Train using L2SGD algorithm
    fn train_l2sgd(&mut self, fgen: &mut FeatureGenerator) -> io::Result<()> {
        use rand::SeedableRng;
        use rand::seq::SliceRandom;

        let num_features = fgen.num_features();
        let num_labels = self.labels.len();
        let max_items = self
            .instances
            .iter()
            .map(|inst| inst.num_items as usize)
            .max()
            .unwrap_or(0);

        let mut weights = vec![0.0; num_features];
        let num_instances = self.instances.len();
        let lambda = 2.0 * self.params.c2 / num_instances as f64;

        // Create CRF context
        let mut ctx = CrfContext::new(num_labels, max_items);

        if self.params.verbose {
            println!("Training with L2SGD (c2={})...", self.params.c2);
        }

        let mut rng = match self.params.shuffle_seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };

        // Calibration phase: find optimal learning rate
        let t0 = self.calibrate_learning_rate(fgen, &mut ctx, lambda, &mut rng)?;

        if self.params.verbose {
            let eta = 1.0 / (lambda * t0);
            println!("Calibrated learning rate: {:.6}", eta);
        }

        // Training loop
        let mut indices: Vec<usize> = (0..self.instances.len()).collect();
        let mut objective_history = vec![0.0; self.params.period];
        let mut best_objective = f64::INFINITY;
        let mut best_weights = vec![0.0; num_features];
        let mut t = 0.0f64;

        for epoch in 1..=self.params.max_iterations {
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
                return Err(io::Error::new(io::ErrorKind::Other, "L2SGD overflow loss"));
            }

            // Include the L2 norm of feature weights to the objective.
            let norm2: f64 = weights.iter().map(|w| w * w).sum();
            sum_loss += 0.5 * lambda * norm2 * num_instances as f64;

            if self.params.verbose {
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

            let improvement = if epoch > self.params.period {
                let prev = objective_history[(epoch - 1) % self.params.period];
                (prev - sum_loss) / sum_loss
            } else {
                self.params.delta
            };

            objective_history[(epoch - 1) % self.params.period] = sum_loss;

            if self.params.verbose && epoch > self.params.period {
                println!("Improvement ratio: {:.6}", improvement);
            }

            if epoch > self.params.period && improvement < self.params.delta {
                if self.params.verbose {
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
        ctx: &mut CrfContext,
        lambda: f64,
        rng: &mut StdRng,
    ) -> io::Result<f64> {
        use rand::seq::SliceRandom;

        let num_features = fgen.num_features();
        let num_instances = self.instances.len();

        // Select calibration samples
        let num_samples = self.params.calibration_samples.min(num_instances);
        let mut sample_indices: Vec<usize> = (0..num_instances).collect();
        sample_indices.shuffle(rng);
        sample_indices.truncate(num_samples);

        let mut eta = self.params.calibration_eta;
        let mut best_eta = eta;
        let mut best_loss = f64::INFINITY;
        let mut dec = false;
        let mut num = self.params.calibration_candidates;
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
                    eta *= self.params.calibration_rate;
                } else {
                    dec = true;
                    num = self.params.calibration_candidates;
                    eta = self.params.calibration_eta / self.params.calibration_rate;
                }
            } else {
                eta /= self.params.calibration_rate;
            }

            trials += 1;
            if self.params.calibration_max_trials <= trials {
                break;
            }
        }

        Ok(1.0 / (lambda * best_eta))
    }
}

impl Default for Trainer {
    fn default() -> Self {
        Self::new(Algorithm::LBFGS)
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
        let mut trainer = Trainer::new(Algorithm::LBFGS);

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
        let mut trainer = Trainer::new(Algorithm::LBFGS);
        assert!(trainer.set("c1", "0.5").is_ok());
        assert!(trainer.set("c2", "2.0").is_ok());
        assert_eq!(trainer.get("c1").unwrap(), "0.5");
        assert_eq!(trainer.get("c2").unwrap(), "2");

        assert!(trainer.set("invalid_param", "1.0").is_err());
        assert!(trainer.get("invalid_param").is_err());
    }

    #[test]
    fn test_trainer_rejects_empty_sequences() {
        let mut trainer = Trainer::new(Algorithm::LBFGS);

        // Empty sequences should be rejected
        let xseq: Vec<Vec<Attribute>> = vec![];
        let yseq: Vec<&str> = vec![];

        let result = trainer.append(&xseq, &yseq);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("empty"));
    }
}
