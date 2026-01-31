use super::feature_gen::{FeatureGenerator, FeatureType};
use crate::dataset::Instance;

/// CRF context for computing forward-backward algorithm
pub struct CrfContext {
    /// Number of labels
    num_labels: usize,
    /// State scores [time][label]
    state_scores: Vec<Vec<f64>>,
    /// Transition scores [prev_label][label]
    trans_scores: Vec<Vec<f64>>,
    /// Forward variables (in log space) [time][label]
    alpha: Vec<Vec<f64>>,
    /// Backward variables (in log space) [time][label]
    beta: Vec<Vec<f64>>,
    /// Marginal probabilities [time][label]
    marginals: Vec<Vec<f64>>,
    /// Transition marginals [time][prev_label][label]
    trans_marginals: Vec<Vec<Vec<f64>>>,
    /// Reusable buffer for log-sum-exp computations
    log_buffer: Vec<f64>,
}

impl CrfContext {
    /// Create a new CRF context
    pub fn new(num_labels: usize, max_items: usize) -> Self {
        Self {
            num_labels,
            state_scores: vec![vec![0.0; num_labels]; max_items],
            trans_scores: vec![vec![0.0; num_labels]; num_labels],
            alpha: vec![vec![f64::NEG_INFINITY; num_labels]; max_items],
            beta: vec![vec![f64::NEG_INFINITY; num_labels]; max_items],
            marginals: vec![vec![0.0; num_labels]; max_items],
            trans_marginals: vec![vec![vec![0.0; num_labels]; num_labels]; max_items],
            log_buffer: vec![0.0; num_labels],
        }
    }

    /// Compute state and transition scores for an instance
    pub fn compute_scores(&mut self, inst: &Instance, fgen: &FeatureGenerator) {
        let seq_len = inst.num_items as usize;

        // Reset scores
        for t in 0..seq_len {
            for l in 0..self.num_labels {
                self.state_scores[t][l] = 0.0;
            }
        }
        for i in 0..self.num_labels {
            for j in 0..self.num_labels {
                self.trans_scores[i][j] = 0.0;
            }
        }

        // Compute state scores
        for t in 0..seq_len {
            for attr in &inst.items[t] {
                let aid = attr.id as usize;
                if aid < fgen.attr_refs.len() {
                    for &fid in &fgen.attr_refs[aid].fids {
                        let feature = &fgen.features[fid as usize];
                        if feature.ftype == FeatureType::State {
                            let lid = feature.dst as usize;
                            self.state_scores[t][lid] += feature.weight * attr.value;
                        }
                    }
                }
            }
        }

        // Compute transition scores
        for l in 0..self.num_labels {
            if l < fgen.label_refs.len() {
                for &fid in &fgen.label_refs[l].fids {
                    let feature = &fgen.features[fid as usize];
                    if feature.ftype == FeatureType::Transition {
                        let prev_lid = feature.src as usize;
                        let lid = feature.dst as usize;
                        self.trans_scores[prev_lid][lid] += feature.weight;
                    }
                }
            }
        }
    }

    /// Log-sum-exp trick for numerical stability.
    ///
    /// Computes log(sum(exp(values))) in a numerically stable way.
    /// Returns NEG_INFINITY for empty arrays or arrays where all values are NEG_INFINITY.
    fn logsumexp(values: &[f64]) -> f64 {
        if values.is_empty() {
            return f64::NEG_INFINITY;
        }
        let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if max_val.is_infinite() {
            return max_val;
        }
        let sum: f64 = values.iter().map(|&v| (v - max_val).exp()).sum();
        max_val + sum.ln()
    }

    /// Forward algorithm in log space
    pub fn forward(&mut self, seq_len: usize) -> f64 {
        // Initialize at t=0
        for l in 0..self.num_labels {
            self.alpha[0][l] = self.state_scores[0][l];
        }

        // Forward recursion - reuse log_buffer to avoid allocations
        for t in 1..seq_len {
            for l in 0..self.num_labels {
                for prev_l in 0..self.num_labels {
                    self.log_buffer[prev_l] = self.alpha[t - 1][prev_l]
                        + self.trans_scores[prev_l][l]
                        + self.state_scores[t][l];
                }
                self.alpha[t][l] = Self::logsumexp(&self.log_buffer[..self.num_labels]);
            }
        }

        // Compute log partition function
        for l in 0..self.num_labels {
            self.log_buffer[l] = self.alpha[seq_len - 1][l];
        }
        Self::logsumexp(&self.log_buffer[..self.num_labels])
    }

    /// Backward algorithm in log space
    pub fn backward(&mut self, seq_len: usize) {
        // Initialize at t=T-1
        for l in 0..self.num_labels {
            self.beta[seq_len - 1][l] = 0.0; // log(1) = 0
        }

        // Backward recursion - reuse log_buffer to avoid allocations
        for t in (0..seq_len - 1).rev() {
            for l in 0..self.num_labels {
                for next_l in 0..self.num_labels {
                    self.log_buffer[next_l] = self.beta[t + 1][next_l]
                        + self.trans_scores[l][next_l]
                        + self.state_scores[t + 1][next_l];
                }
                self.beta[t][l] = Self::logsumexp(&self.log_buffer[..self.num_labels]);
            }
        }
    }

    /// Compute marginal probabilities
    pub fn compute_marginals(&mut self, seq_len: usize, log_z: f64) {
        // State marginals
        for t in 0..seq_len {
            for l in 0..self.num_labels {
                let log_marginal = self.alpha[t][l] + self.beta[t][l] - log_z;
                self.marginals[t][l] = log_marginal.exp();
            }
        }

        // Transition marginals
        for t in 1..seq_len {
            for prev_l in 0..self.num_labels {
                for l in 0..self.num_labels {
                    let log_marginal = self.alpha[t - 1][prev_l]
                        + self.trans_scores[prev_l][l]
                        + self.state_scores[t][l]
                        + self.beta[t][l]
                        - log_z;
                    self.trans_marginals[t][prev_l][l] = log_marginal.exp();
                }
            }
        }
    }

    /// Compute expected feature counts into a pre-allocated vector
    pub fn expected_counts_into(
        &self,
        inst: &Instance,
        fgen: &FeatureGenerator,
        counts: &mut [f64],
    ) {
        let seq_len = inst.num_items as usize;

        // State feature expectations
        for t in 0..seq_len {
            for attr in &inst.items[t] {
                let aid = attr.id as usize;
                if aid < fgen.attr_refs.len() {
                    for &fid in &fgen.attr_refs[aid].fids {
                        let feature = &fgen.features[fid as usize];
                        if feature.ftype == FeatureType::State {
                            let lid = feature.dst as usize;
                            counts[fid as usize] += self.marginals[t][lid] * attr.value;
                        }
                    }
                }
            }
        }

        // Transition feature expectations
        for t in 1..seq_len {
            for prev_l in 0..self.num_labels {
                if prev_l < fgen.label_refs.len() {
                    for &fid in &fgen.label_refs[prev_l].fids {
                        let feature = &fgen.features[fid as usize];
                        if feature.ftype == FeatureType::Transition {
                            let lid = feature.dst as usize;
                            counts[fid as usize] += self.trans_marginals[t][prev_l][lid];
                        }
                    }
                }
            }
        }
    }

    /// Compute observed feature counts into a pre-allocated vector
    pub fn observed_counts_into(
        &self,
        inst: &Instance,
        fgen: &FeatureGenerator,
        counts: &mut [f64],
    ) {
        let seq_len = inst.num_items as usize;

        // State feature observations
        for t in 0..seq_len {
            let label_id = inst.labels[t]; // u32 label ID for this timestep
            for attr in &inst.items[t] {
                let aid = attr.id as usize;
                if aid < fgen.attr_refs.len() {
                    for &fid in &fgen.attr_refs[aid].fids {
                        let feature = &fgen.features[fid as usize];
                        // feature.dst is the target label ID for this state feature
                        if feature.ftype == FeatureType::State && feature.dst == label_id {
                            counts[fid as usize] += attr.value;
                        }
                    }
                }
            }
        }

        // Transition feature observations
        for t in 1..seq_len {
            let prev_label = inst.labels[t - 1];
            let label = inst.labels[t];
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
    }

    /// Compute log-likelihood for an instance using pre-computed scores.
    ///
    /// This method assumes that [`compute_scores`](Self::compute_scores) has already been
    /// called and that `forward()` has been run to populate `self.alpha`.
    /// It computes the score of the correct label sequence and subtracts
    /// the partition function (log_z) to get the log-likelihood.
    ///
    /// # Arguments
    ///
    /// * `inst` - The training instance
    /// * `log_z` - The log partition function from the forward algorithm
    pub fn log_likelihood(&self, inst: &Instance, log_z: f64) -> f64 {
        let seq_len = inst.num_items as usize;

        // Compute score of the correct label sequence
        let mut score = 0.0;
        for t in 0..seq_len {
            let label = inst.labels[t] as usize;
            score += self.state_scores[t][label];
            if t > 0 {
                let prev_label = inst.labels[t - 1] as usize;
                score += self.trans_scores[prev_label][label];
            }
        }

        score - log_z
    }
}
