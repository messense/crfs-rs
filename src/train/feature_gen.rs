use std::collections::BTreeMap;
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
}

/// Feature references for fast lookup
#[derive(Debug, Clone, Default)]
pub struct FeatureRefs {
    /// Feature IDs
    pub fids: Vec<u32>,
}

/// Feature generator for CRF training
pub struct FeatureGenerator {
    /// All features
    pub features: Vec<Feature>,
    /// Feature references by attribute ID
    pub attr_refs: Vec<FeatureRefs>,
    /// Feature references by label ID (for transitions)
    pub label_refs: Vec<FeatureRefs>,
}

impl FeatureGenerator {
    /// Generate features from training instances.
    ///
    /// Features with frequency >= `min_freq` are included. Features with frequency
    /// exactly equal to `min_freq` are included. The default `min_freq` of 0.0
    /// ensures all features that occur at least once are included.
    pub fn generate(
        instances: &[Instance],
        attrs: &Dictionary,
        labels: &Dictionary,
        min_freq: f64,
    ) -> io::Result<Self> {
        let num_labels = labels.len();
        let num_attrs = attrs.len();

        // Count feature occurrences
        let mut state_counts: BTreeMap<(u32, u32, u32), f64> = BTreeMap::new();
        let mut trans_counts: BTreeMap<(u32, u32, u32), f64> = BTreeMap::new();

        for inst in instances {
            let seq_len = inst.num_items as usize;
            let inst_weight = inst.weight;

            // Count state features
            // State feature frequencies are weighted by attribute values (attr.value)
            // and instance weight, which allows for real-valued feature weights.
            // Transition features are binary (either present or not), so they use
            // 1.0 * inst_weight for each occurrence.
            for t in 0..seq_len {
                let label = inst.labels[t];
                for attr in &inst.items[t] {
                    let key = (FeatureType::State as u32, attr.id, label);
                    *state_counts.entry(key).or_insert(0.0) += attr.value * inst_weight;
                }
            }

            // Count transition features
            for t in 1..seq_len {
                let prev_label = inst.labels[t - 1];
                let label = inst.labels[t];
                let key = (FeatureType::Transition as u32, prev_label, label);
                *trans_counts.entry(key).or_insert(0.0) += inst_weight;
            }
        }

        // Build feature list
        let mut features = Vec::new();
        let mut attr_refs = vec![FeatureRefs::default(); num_attrs];
        let mut label_refs = vec![FeatureRefs::default(); num_labels];

        // Add state features
        for ((_, aid, lid), freq) in state_counts {
            if freq >= min_freq {
                let fid = u32::try_from(features.len())
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "too many features"))?;
                features.push(Feature {
                    ftype: FeatureType::State,
                    src: aid,
                    dst: lid,
                    weight: 0.0,
                });
                attr_refs[aid as usize].fids.push(fid);
            }
        }

        // Add transition features
        for ((_, prev_lid, lid), freq) in trans_counts {
            if freq >= min_freq {
                let fid = u32::try_from(features.len())
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "too many features"))?;
                features.push(Feature {
                    ftype: FeatureType::Transition,
                    src: prev_lid,
                    dst: lid,
                    weight: 0.0,
                });
                label_refs[prev_lid as usize].fids.push(fid);
            }
        }

        Ok(Self {
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
    ///
    /// # Panics
    ///
    /// Panics if `weights.len()` does not equal `self.num_features()`.
    pub fn set_weights(&mut self, weights: &[f64]) {
        assert_eq!(
            weights.len(),
            self.features.len(),
            "weights length ({}) must equal number of features ({})",
            weights.len(),
            self.features.len()
        );
        for (i, feature) in self.features.iter_mut().enumerate() {
            feature.weight = weights[i];
        }
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
