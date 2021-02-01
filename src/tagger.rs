use std::io;

use crate::context::{Context, Flag, Reset};
use crate::dataset::{self, Instance, Item};
use crate::model::Model;

#[derive(Debug, Clone, Copy)]
enum Level {
    None,
    Set,
    AlphaBeta,
}

/// Tuple of attribute and its value
#[derive(Debug, Clone)]
pub struct Attribute {
    /// Attribute name
    pub name: String,
    /// Value of the attribute
    pub value: f64,
}

/// The tagger provides the functionality for predicting label sequences for input sequences using a model
#[derive(Debug, Clone)]
pub struct Tagger<'a> {
    /// CRF model
    model: &'a Model<'a>,
    /// CRF context
    context: Context,
    /// Number of distinct output labels
    num_labels: u32,
    /// Number of distinct attributes
    num_attrs: u32,
    level: Level,
}

impl Attribute {
    pub fn new<T: Into<String>>(name: T, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }
}

impl<'a> Tagger<'a> {
    pub(crate) fn new(model: &'a Model<'a>) -> io::Result<Self> {
        let num_labels = model.num_labels();
        let num_attrs = model.num_attrs();
        let mut context = Context::new(Flag::VITERBI | Flag::MARGINALS, num_labels, 0);
        context.reset(Reset::TRANS);
        let mut tagger = Self {
            model,
            context,
            num_labels,
            num_attrs,
            level: Level::None,
        };
        tagger.transition_score()?;
        tagger.context.exp_transition();
        Ok(tagger)
    }

    /// Obtain the number of items in the current instance.
    pub fn len(&self) -> usize {
        self.context.num_items as usize
    }

    /// Is the number of items in the current instance equals to 0
    pub fn is_empty(&self) -> bool {
        self.context.num_items == 0
    }

    /// Predict the label sequence for the item sequence.
    pub fn tag<T: AsRef<[Attribute]>>(&mut self, xseq: &[T]) -> io::Result<Vec<&str>> {
        if xseq.is_empty() {
            return Ok(Vec::new());
        }
        self.set(xseq)?;
        let (label_ids, _score) = self.viterbi();
        let mut labels = Vec::with_capacity(label_ids.len());
        for id in label_ids {
            let label = self.model.to_label(id).unwrap();
            labels.push(label);
        }
        Ok(labels)
    }

    /// Set an instance (item sequence) for future calls of `tag`, `probability` and `marginal` methods
    pub fn set<T: AsRef<[Attribute]>>(&mut self, xseq: &[T]) -> io::Result<()> {
        let mut instance = Instance::with_capacity(xseq.len());
        for item in xseq {
            let item: Item = item
                .as_ref()
                .iter()
                .filter_map(|x| {
                    self.model
                        .to_attr_id(&x.name)
                        .map(|id| dataset::Attribute::new(id, x.value))
                })
                .collect();
            instance.push(item, 0);
        }
        self.context.set_num_items(instance.num_items);
        self.context.reset(Reset::STATE);
        self.state_score(&instance)?;
        self.level = Level::Set;
        Ok(())
    }

    /// Compute the log of the partition factor (normalization constant).
    pub fn lognorm(&mut self) -> f64 {
        self.set_level(Level::AlphaBeta);
        self.context.log_norm
    }

    /// Compute the score of a label sequence.
    pub fn score(&self, labels: &[u32]) -> f64 {
        self.context.score(labels)
    }

    fn set_level(&mut self, level: Level) {
        let prev = self.level;
        match prev {
            Level::None | Level::Set => {
                self.context.exp_state();
                self.context.alpha_score();
                self.context.beta_score();
            }
            Level::AlphaBeta => {}
        }
        self.level = level;
    }

    fn transition_score(&mut self) -> io::Result<()> {
        // Compute transition scores between two labels
        let l = self.num_labels as usize;
        for i in 0..l {
            let trans = &mut self.context.trans[l * i..];
            let edge = self.model.label_ref(i as u32)?;
            for r in 0..edge.num_features {
                // Transition feature from #i to #(feature.target)
                let fid = edge.get(r as usize)?;
                let feature = self.model.feature(fid)?;
                trans[feature.target as usize] = feature.weight;
            }
        }
        Ok(())
    }

    fn state_score(&mut self, instance: &Instance) -> io::Result<()> {
        // Loop over the items in the sequence
        for t in 0..instance.num_items as usize {
            let item = &instance.items[t];
            let state = &mut self.context.state[self.context.num_labels as usize * t..];
            // Loop over the attributes attached to the item
            for attr in item {
                // Access the list of state features associated with the attribute
                let id = attr.id;
                let attr_ref = self.model.attr_ref(id)?;
                // A scale usually represents the attribute frequency in the item
                let value = attr.value;
                // Loop over the state features associated with the attribue
                for r in 0..attr_ref.num_features as usize {
                    let fid = attr_ref.get(r)?;
                    let feature = self.model.feature(fid)?;
                    state[feature.target as usize] += feature.weight * value;
                }
            }
        }
        Ok(())
    }

    fn viterbi(&mut self) -> (Vec<u32>, f64) {
        self.context.viterbi()
    }
}
