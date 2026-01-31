use std::collections::HashMap;
use std::io;

use super::dictionary::Dictionary;
use crate::dataset::Instance;

/// Feature type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureType {
    /// State feature: (attribute, label) -> weight
    State = 0,
    /// Transition feature: (prev_label, label) -> weight
    Transition = 1,
}

/// A CRF feature descriptor
#[derive(Debug, Clone)]
pub struct Feature {
    /// Feature type
    pub ftype: FeatureType,
    /// Source ID (attribute ID for state, prev label ID for transition)
    pub src: u32,
    /// Target ID (label ID)
    pub dst: u32,
    /// Feature weight
    pub weight: f64,
    /// Feature frequency in training data
    pub freq: f64,
}

/// Feature references for fast lookup
#[derive(Debug, Clone, Default)]
pub struct FeatureRefs {
    /// Feature IDs
    pub fids: Vec<u32>,
}

/// Feature generator for CRF training
pub struct FeatureGenerator {
    /// Number of labels
    pub num_labels: usize,
    /// Number of attributes
    pub num_attrs: usize,
    /// All features
    pub features: Vec<Feature>,
    /// Feature references by attribute ID
    pub attr_refs: Vec<FeatureRefs>,
    /// Feature references by label ID (for transitions)
    pub label_refs: Vec<FeatureRefs>,
}

impl FeatureGenerator {
    /// Generate features from training instances
    pub fn generate(
        instances: &[Instance],
        attrs: &Dictionary,
        labels: &Dictionary,
        min_freq: f64,
    ) -> io::Result<Self> {
        let num_labels = labels.len();
        let num_attrs = attrs.len();

        // Count feature occurrences
        let mut state_counts: HashMap<(u32, u32), f64> = HashMap::new();
        let mut trans_counts: HashMap<(u32, u32), f64> = HashMap::new();

        for inst in instances {
            let seq_len = inst.num_items as usize;

            // Count state features
            for t in 0..seq_len {
                let label = inst.labels[t];
                for attr in &inst.items[t] {
                    let key = (attr.id, label);
                    *state_counts.entry(key).or_insert(0.0) += attr.value;
                }
            }

            // Count transition features
            for t in 1..seq_len {
                let prev_label = inst.labels[t - 1];
                let label = inst.labels[t];
                let key = (prev_label, label);
                *trans_counts.entry(key).or_insert(0.0) += 1.0;
            }
        }

        // Build feature list
        let mut features = Vec::new();
        let mut attr_refs = vec![FeatureRefs::default(); num_attrs];
        let mut label_refs = vec![FeatureRefs::default(); num_labels];

        // Add state features
        for ((aid, lid), freq) in state_counts {
            if freq >= min_freq {
                let fid = features.len() as u32;
                features.push(Feature {
                    ftype: FeatureType::State,
                    src: aid,
                    dst: lid,
                    weight: 0.0,
                    freq,
                });
                attr_refs[aid as usize].fids.push(fid);
            }
        }

        // Add transition features
        for ((prev_lid, lid), freq) in trans_counts {
            if freq >= min_freq {
                let fid = features.len() as u32;
                features.push(Feature {
                    ftype: FeatureType::Transition,
                    src: prev_lid,
                    dst: lid,
                    weight: 0.0,
                    freq,
                });
                label_refs[prev_lid as usize].fids.push(fid);
            }
        }

        Ok(Self {
            num_labels,
            num_attrs,
            features,
            attr_refs,
            label_refs,
        })
    }

    /// Get the number of features
    pub fn num_features(&self) -> usize {
        self.features.len()
    }

    /// Update feature weights from a weight vector
    pub fn set_weights(&mut self, weights: &[f64]) {
        for (i, feature) in self.features.iter_mut().enumerate() {
            if i < weights.len() {
                feature.weight = weights[i];
            }
        }
    }

    /// Get feature weights as a vector
    pub fn get_weights(&self) -> Vec<f64> {
        self.features.iter().map(|f| f.weight).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Attribute;

    #[test]
    fn test_feature_generation() {
        let mut attrs = Dictionary::new();
        let mut labels = Dictionary::new();

        let walk_id = attrs.get_or_insert("walk");
        let shop_id = attrs.get_or_insert("shop");
        let sunny_id = labels.get_or_insert("sunny");
        let rainy_id = labels.get_or_insert("rainy");

        let mut inst = Instance::with_capacity(3);
        inst.push(vec![Attribute::new(walk_id, 1.0)], sunny_id);
        inst.push(vec![Attribute::new(shop_id, 1.0)], sunny_id);
        inst.push(vec![Attribute::new(walk_id, 1.0)], rainy_id);

        let instances = vec![inst];
        let fgen = FeatureGenerator::generate(&instances, &attrs, &labels, 0.0).unwrap();

        // Should have state features and transition features
        assert!(fgen.num_features() > 0);
        assert_eq!(fgen.num_labels, 2);
        assert_eq!(fgen.num_attrs, 2);

        // Check that we have both state and transition features
        let has_state = fgen.features.iter().any(|f| f.ftype == FeatureType::State);
        let has_trans = fgen
            .features
            .iter()
            .any(|f| f.ftype == FeatureType::Transition);
        assert!(has_state);
        assert!(has_trans);
    }
}
