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
