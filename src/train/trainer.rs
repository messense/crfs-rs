use std::io;
use std::path::Path;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use super::dictionary::Dictionary;
use super::feature_gen::FeatureGenerator;
use super::model_writer::ModelWriter;
use crate::attribute::Attribute;
use crate::dataset::{Attribute as DatasetAttribute, Instance};

mod arow;
mod averaged_perceptron;
mod l2sgd;
mod lbfgs;
mod passive_aggressive;

pub use self::arow::ArowParams;
pub use self::averaged_perceptron::AveragedPerceptronParams;
pub use self::l2sgd::L2SgdParams;
pub use self::lbfgs::{LbfgsParams, LineSearchAlgorithm};
pub use self::passive_aggressive::{PaType, PassiveAggressiveParams};

fn shuffle_indices(indices: &mut [usize], rng: &mut StdRng) {
    indices.shuffle(rng);
}

/// Training algorithm marker for L-BFGS.
#[derive(Debug, Clone, Copy)]
pub struct Lbfgs;

/// Training algorithm marker for Averaged Perceptron.
#[derive(Debug, Clone, Copy)]
pub struct AveragedPerceptron;

/// Training algorithm marker for Passive Aggressive.
#[derive(Debug, Clone, Copy)]
pub struct PassiveAggressive;

/// Training algorithm marker for L2SGD.
#[derive(Debug, Clone, Copy)]
pub struct L2Sgd;

/// Training algorithm marker for AROW.
#[derive(Debug, Clone, Copy)]
pub struct Arow;

/// Training algorithm interface.
pub trait TrainingAlgorithm {
    type Params: Default;

    fn train(trainer: &mut Trainer<Self>, fgen: &mut FeatureGenerator) -> io::Result<()>
    where
        Self: Sized;
}

/// CRF Trainer
#[derive(Debug)]
pub struct Trainer<A: TrainingAlgorithm> {
    /// Training instances
    instances: Vec<Instance>,
    /// Attribute dictionary
    attrs: Dictionary,
    /// Label dictionary
    labels: Dictionary,
    /// Minimum feature frequency
    feature_minfreq: f64,
    /// Enable verbose output
    verbose: bool,
    /// Training parameters
    params: A::Params,
}

impl<A: TrainingAlgorithm> Trainer<A> {
    /// Create a new trainer
    pub fn new() -> Self {
        Self {
            instances: Vec::new(),
            attrs: Dictionary::new(),
            labels: Dictionary::new(),
            feature_minfreq: 0.0,
            verbose: false,
            params: A::Params::default(),
        }
    }

    /// Enable or disable verbose output
    pub fn verbose(&mut self, enabled: bool) -> &mut Self {
        self.verbose = enabled;
        self
    }

    /// Get training parameters
    pub fn params(&self) -> &A::Params {
        &self.params
    }

    /// Get training parameters for mutation
    pub fn params_mut(&mut self) -> &mut A::Params {
        &mut self.params
    }

    /// Get minimum feature frequency
    pub fn feature_minfreq(&self) -> f64 {
        self.feature_minfreq
    }

    /// Set minimum feature frequency
    pub fn set_feature_minfreq(&mut self, feature_minfreq: f64) -> io::Result<()> {
        if feature_minfreq < 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "feature.minfreq must be non-negative",
            ));
        }
        self.feature_minfreq = feature_minfreq;
        Ok(())
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

    /// Train the model and save to file
    pub fn train(&mut self, filename: &Path) -> io::Result<()> {
        if self.instances.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "no training data",
            ));
        }

        // Generate features
        if self.verbose {
            println!("Generating features...");
        }
        let mut fgen = FeatureGenerator::generate(
            &self.instances,
            &self.attrs,
            &self.labels,
            self.feature_minfreq,
        )?;

        if self.verbose {
            println!("Number of features: {}", fgen.num_features());
            println!("Number of labels: {}", self.labels.len());
            println!("Number of attributes: {}", self.attrs.len());
        }

        // Train with selected algorithm
        A::train(self, &mut fgen)?;

        // Save model
        if self.verbose {
            println!("Saving model to {}...", filename.display());
        }
        ModelWriter::write(filename, &fgen, &self.labels, &self.attrs)?;

        if self.verbose {
            println!("Training completed.");
        }

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
        for (t, &label) in labels.iter().enumerate().take(seq_len) {
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

impl Trainer<Lbfgs> {
    /// Create a new L-BFGS trainer
    pub fn lbfgs() -> Self {
        Self::new()
    }

    /// Set L1 regularization coefficient (builder pattern)
    pub fn with_c1(mut self, c1: f64) -> io::Result<Self> {
        self.params.set_c1(c1)?;
        Ok(self)
    }

    /// Set L2 regularization coefficient (builder pattern)
    pub fn with_c2(mut self, c2: f64) -> io::Result<Self> {
        self.params.set_c2(c2)?;
        Ok(self)
    }

    /// Set maximum iterations (builder pattern)
    pub fn with_max_iterations(mut self, max_iterations: usize) -> io::Result<Self> {
        self.params.set_max_iterations(max_iterations)?;
        Ok(self)
    }

    /// Set convergence epsilon (builder pattern)
    pub fn with_epsilon(mut self, epsilon: f64) -> io::Result<Self> {
        self.params.set_epsilon(epsilon)?;
        Ok(self)
    }
}

impl Trainer<AveragedPerceptron> {
    /// Create a new Averaged Perceptron trainer
    pub fn averaged_perceptron() -> Self {
        Self::new()
    }

    /// Set maximum iterations (builder pattern)
    pub fn with_max_iterations(mut self, max_iterations: usize) -> io::Result<Self> {
        self.params.set_max_iterations(max_iterations)?;
        Ok(self)
    }

    /// Set convergence epsilon (builder pattern)
    pub fn with_epsilon(mut self, epsilon: f64) -> io::Result<Self> {
        self.params.set_epsilon(epsilon)?;
        Ok(self)
    }
}

impl Trainer<PassiveAggressive> {
    /// Create a new Passive Aggressive trainer
    pub fn passive_aggressive() -> Self {
        Self::new()
    }

    /// Set PA type (builder pattern)
    pub fn with_pa_type(mut self, pa_type: PaType) -> Self {
        self.params.set_pa_type(pa_type);
        self
    }

    /// Set aggressiveness parameter C (builder pattern)
    pub fn with_c(mut self, c: f64) -> io::Result<Self> {
        self.params.set_pa_c(c)?;
        Ok(self)
    }

    /// Set error sensitivity (builder pattern)
    pub fn with_error_sensitive(mut self, enabled: bool) -> Self {
        self.params.set_pa_error_sensitive(enabled);
        self
    }

    /// Set weight averaging (builder pattern)
    pub fn with_averaging(mut self, enabled: bool) -> Self {
        self.params.set_pa_averaging(enabled);
        self
    }

    /// Set maximum iterations (builder pattern)
    pub fn with_max_iterations(mut self, max_iterations: usize) -> io::Result<Self> {
        self.params.set_max_iterations(max_iterations)?;
        Ok(self)
    }

    /// Set convergence epsilon (builder pattern)
    pub fn with_epsilon(mut self, epsilon: f64) -> io::Result<Self> {
        self.params.set_epsilon(epsilon)?;
        Ok(self)
    }
}

impl Trainer<L2Sgd> {
    /// Create a new L2-regularized SGD trainer
    pub fn l2sgd() -> Self {
        Self::new()
    }

    /// Set L2 regularization coefficient (builder pattern)
    pub fn with_c2(mut self, c2: f64) -> io::Result<Self> {
        self.params.set_c2(c2)?;
        Ok(self)
    }

    /// Set maximum iterations (builder pattern)
    pub fn with_max_iterations(mut self, max_iterations: usize) -> io::Result<Self> {
        self.params.set_max_iterations(max_iterations)?;
        Ok(self)
    }

    /// Set convergence check period (builder pattern)
    pub fn with_period(mut self, period: usize) -> io::Result<Self> {
        self.params.set_period(period)?;
        Ok(self)
    }

    /// Set convergence delta threshold (builder pattern)
    pub fn with_delta(mut self, delta: f64) -> io::Result<Self> {
        self.params.set_delta(delta)?;
        Ok(self)
    }
}

impl Trainer<Arow> {
    /// Create a new AROW trainer
    pub fn arow() -> Self {
        Self::new()
    }

    /// Set initial variance (builder pattern)
    pub fn with_variance(mut self, variance: f64) -> io::Result<Self> {
        self.params.set_variance(variance)?;
        Ok(self)
    }

    /// Set regularization parameter gamma (builder pattern)
    pub fn with_gamma(mut self, gamma: f64) -> io::Result<Self> {
        self.params.set_gamma(gamma)?;
        Ok(self)
    }

    /// Set maximum iterations (builder pattern)
    pub fn with_max_iterations(mut self, max_iterations: usize) -> io::Result<Self> {
        self.params.set_max_iterations(max_iterations)?;
        Ok(self)
    }

    /// Set convergence epsilon (builder pattern)
    pub fn with_epsilon(mut self, epsilon: f64) -> io::Result<Self> {
        self.params.set_epsilon(epsilon)?;
        Ok(self)
    }
}

impl<A: TrainingAlgorithm> Default for Trainer<A> {
    fn default() -> Self {
        Self::new()
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
        let mut trainer = Trainer::lbfgs();

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
        let mut trainer = Trainer::lbfgs();
        assert!(trainer.params_mut().set_c1(0.5).is_ok());
        assert!(trainer.params_mut().set_c2(2.0).is_ok());
        assert_eq!(trainer.params().c1(), 0.5);
        assert_eq!(trainer.params().c2(), 2.0);
    }

    #[test]
    fn test_trainer_rejects_empty_sequences() {
        let mut trainer = Trainer::lbfgs();

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
