use bitflags::bitflags;
use ndarray::{Array1, Array2, s};

bitflags! {
    /// Functionality flags for contexts
    #[derive(Default)]
    pub struct Flag: u32 {
        const BASE = 0x01;
        const VITERBI = 0x01;
        const MARGINALS = 0x02;
        const ALL = 0xFF;
    }
}

bitflags! {
    /// Reset flags
    pub struct Reset: u32 {
        /// Reset state scores
        const STATE = 0x01;
        /// Reset transition scores
        const TRANS = 0x02;
        /// Reset all
        const ALL = 0xFF;
    }
}

/// Context maintains internal data for an instance
#[derive(Debug, Clone)]
pub struct Context {
    /// Flag specifying the functionality
    flag: Flag,
    /// The total number of distinct labels
    pub num_labels: u32,
    /// The number of items in the instance
    pub num_items: u32,
    /// The maximum number of labels
    cap_items: u32,
    /// Logarithm of the normalization factor for the instance.
    ///
    /// This is equivalent to the total scores of all paths in the lattice.
    log_norm: f64,
    /// State scores
    ///
    /// This is a `[T][L]` matrix whose element `[t][l]` presents total score
    /// of state features associating label #l at #t.
    pub state: Array2<f64>,
    /// Transition scores
    ///
    /// This is a `[L][L]` matrix whose element `[i][j]` represents the total
    /// score of transition features associating labels #i and #j.
    pub trans: Array2<f64>,
    /// Alpha score matrix
    ///
    /// This is a `[T][L]` matrix whose element `[t][l]` presents the total
    /// score of paths starting at BOS and arriving at (t, l).
    alpha_score: Array2<f64>,
    /// Beta score matrix
    ///
    /// This is a `[T][L]` matrix whose element `[t][l]` presents the total
    /// score of paths starting at (t, l) and arriving at EOS.
    beta_score: Array2<f64>,
    /// Scale factor vector
    ///
    /// This is a `[T]` vector whose element `[t]` presents the scaling
    /// coefficient for the alpha_score and beta_score.
    scale_factor: Array1<f64>,
    /// Row vector (work space)
    ///
    /// This is a `[L]` vector used internally for a work space.
    row: Array1<f64>,
    /// Backward edges
    ///
    /// This is a `[T][L]` matrix whose element `[t][j]` represents the label #i
    /// that yields the maximum score to arrive at (t, j).
    /// This member is available only with `CTXF_VITERBI` flag enabled.
    backward_edge: Array2<u32>,
    /// Exponents of state scores
    ///
    /// This is a `[T][L]` matrix whose element `[t][l]` presents the exponent
    /// of the total score of state features associating label #l at #t.
    /// This member is available only with `CTXF_MARGINALS` flag.
    exp_state: Array2<f64>,
    /// Exponents of transition scores.
    ///
    /// This is a `[L][L]` matrix whose element `[i][j]` represents the exponent
    /// of the total score of transition features associating labels #i and #j.
    /// This member is available only with `CTXF_MARGINALS` flag.
    exp_trans: Array2<f64>,
    /// Model expectations of states.
    ///
    /// This is a `[T][L]` matrix whose element `[t][l]` presents the model
    /// expectation (marginal probability) of the state (t,l)
    /// This member is available only with CTXF_MARGINALS flag.
    mexp_state: Array2<f64>,
    /// Model expectations of transitions.
    ///
    /// This is a `[L][L]` matrix whose element `[i][j]` presents the model
    /// expectation of the transition (i--j).
    /// This member is available only with `CTXF_MARGINALS` flag.
    mexp_trans: Array2<f64>,
}

impl Default for Context {
    fn default() -> Self {
        Self {
            flag: Flag::default(),
            num_labels: 0,
            num_items: 0,
            cap_items: 0,
            log_norm: 0.0,
            state: Array2::zeros((0, 0)),
            trans: Array2::zeros((0, 0)),
            alpha_score: Array2::zeros((0, 0)),
            beta_score: Array2::zeros((0, 0)),
            scale_factor: Array1::zeros(0),
            row: Array1::zeros(0),
            backward_edge: Array2::zeros((0, 0)),
            exp_state: Array2::zeros((0, 0)),
            exp_trans: Array2::zeros((0, 0)),
            mexp_state: Array2::zeros((0, 0)),
            mexp_trans: Array2::zeros((0, 0)),
        }
    }
}

impl Context {
    pub fn new(flag: Flag, l: u32, t: u32) -> Self {
        let l = l as usize;
        let t = t as usize;
        
        let trans = Array2::zeros((l, l));
        let (exp_trans, mexp_trans) = if flag.contains(Flag::MARGINALS) {
            (Array2::zeros((l, l)), Array2::zeros((l, l)))
        } else {
            (Array2::zeros((0, 0)), Array2::zeros((0, 0)))
        };
        
        let mut ctx = Self {
            flag,
            trans,
            exp_trans,
            mexp_trans,
            num_items: 0,
            num_labels: l as u32,
            ..Default::default()
        };
        ctx.set_num_items(t as u32);
        // t gives the 'hint' for maximum length of items.
        ctx.num_items = 0;
        ctx
    }

    pub fn set_num_items(&mut self, t: u32) {
        self.num_items = t;
        if self.cap_items < t {
            let l = self.num_labels as usize;
            let t = t as usize;
            
            self.alpha_score = Array2::zeros((t, l));
            self.beta_score = Array2::zeros((t, l));
            self.scale_factor = Array1::zeros(t);
            self.row = Array1::zeros(l);
            
            if self.flag.contains(Flag::VITERBI) {
                self.backward_edge = Array2::zeros((t, l));
            }
            
            self.state = Array2::zeros((t, l));
            
            if self.flag.contains(Flag::MARGINALS) {
                self.exp_state = Array2::zeros((t, l));
                self.mexp_state = Array2::zeros((t, l));
            }
            
            self.cap_items = t as u32;
        }
    }

    pub fn reset(&mut self, flag: Reset) {
        let t = self.num_items as usize;
        let l = self.num_labels as usize;
        
        if flag.contains(Reset::STATE) && t > 0 && l > 0 {
            self.state.slice_mut(s![..t, ..l]).fill(0.0);
        }
        if flag.contains(Reset::TRANS) {
            self.trans.fill(0.0);
        }
        if self.flag.contains(Flag::MARGINALS) {
            if t > 0 && l > 0 {
                self.mexp_state.slice_mut(s![..t, ..l]).fill(0.0);
            }
            self.mexp_trans.fill(0.0);
            self.log_norm = 0.0;
        }
    }

    pub fn exp_transition(&mut self) {
        self.exp_trans.assign(&self.trans);
        self.exp_trans.mapv_inplace(|x| x.exp());
    }

    pub fn viterbi(&mut self) -> (Vec<u32>, f64) {
        let l = self.num_labels as usize;
        let t = self.num_items as usize;
        
        // Compute the scores at (0, *)
        for j in 0..l {
            self.alpha_score[[0, j]] = self.state[[0, j]];
        }
        
        // Compute the scores at (t, *)
        for time in 1..t {
            // Compute the score of (t, j)
            for j in 0..l {
                let mut max_score = f64::MIN;
                let mut argmax_score = 0;
                
                for i in 0..l {
                    // Transit from (t-1, i) to (t, j)
                    let score = self.alpha_score[[time - 1, i]] + self.trans[[i, j]];
                    
                    // Store this path if it has the maximum score
                    if max_score < score {
                        max_score = score;
                        argmax_score = i;
                    }
                }
                
                // Backward link (#t, #j) -> (#t-1, #i)
                self.backward_edge[[time, j]] = argmax_score as u32;
                
                // Add the state score on (t, j)
                self.alpha_score[[time, j]] = max_score + self.state[[time, j]];
            }
        }
        
        // Find the node (#T, Ei) that reaches EOS with the maximum score
        let mut max_score = f64::MIN;
        let last_scores = self.alpha_score.row(t - 1);
        let mut labels = vec![0u32; t];
        
        for (i, &score) in last_scores.iter().enumerate() {
            if max_score < score {
                max_score = score;
                labels[t - 1] = i as u32;
            }
        }
        
        // Tag labels by tracing the backward links
        for time in (0..t - 1).rev() {
            let next_label = labels[time + 1] as usize;
            labels[time] = self.backward_edge[[time + 1, next_label]];
        }
        
        (labels, max_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_new() {
        let _ctx = Context::new(Flag::VITERBI, 2, 0);
        let _ctx = Context::new(Flag::MARGINALS, 2, 0);
        let _ctx = Context::new(Flag::VITERBI | Flag::MARGINALS, 2, 0);
    }

    #[test]
    fn test_context_reset() {
        let mut ctx = Context::new(Flag::VITERBI | Flag::MARGINALS, 2, 0);
        ctx.reset(Reset::STATE);
        ctx.reset(Reset::TRANS);
        ctx.reset(Reset::STATE | Reset::TRANS);
    }
}
