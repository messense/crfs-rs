use bitflags::bitflags;

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
#[derive(Debug, Clone, Default)]
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
    pub state: Vec<f64>,
    /// Transition scores
    ///
    /// This is a `[L][L]` matrix whose element `[i][j]` represents the total
    /// score of transition features associating labels #i and #j.
    pub trans: Vec<f64>,
    /// Transposed transition scores for cache-friendly Viterbi
    ///
    /// This is a `[L][L]` matrix whose element `[j][i]` = trans[i][j].
    /// Stored for optimized column-wise access during Viterbi.
    pub(crate) trans_t: Vec<f64>,
    /// Alpha score matrix
    ///
    /// This is a `[T][L]` matrix whose element `[t][l]` presents the total
    /// score of paths starting at BOS and arriving at (t, l).
    alpha_score: Vec<f64>,
    /// Beta score matrix
    ///
    /// This is a `[T][L]` matrix whose element `[t][l]` presents the total
    /// score of paths starting at (t, l) and arriving at EOS.
    beta_score: Vec<f64>,
    /// Scale factor vector
    ///
    /// This is a `[T]` vector whose element `[t]` presents the scaling
    /// coefficient for the alpha_score and beta_score.
    scale_factor: Vec<f64>,
    /// Row vector (work space)
    ///
    /// This is a `[T]` vector used internally for a work space.
    row: Vec<f64>,
    /// Backward edges
    ///
    /// This is a `[T][L]` matrix whose element `[t][j]` represents the label #i
    /// that yields the maximum score to arrive at (t, j).
    /// This member is available only with `CTXF_VITERBI` flag enabled.
    backward_edge: Vec<u32>,
    /// Exponents of state scores
    ///
    /// This is a `[T][L]` matrix whose element `[t][l]` presents the exponent
    /// of the total score of state features associating label #l at #t.
    /// This member is available only with `CTXF_MARGINALS` flag.
    exp_state: Vec<f64>,
    /// Exponents of transition scores.
    ///
    /// This is a `[L][L]` matrix whose element `[i][j]` represents the exponent
    /// of the total score of transition features associating labels #i and #j.
    /// This member is available only with `CTXF_MARGINALS` flag.
    exp_trans: Vec<f64>,
    /// Model expectations of states.
    ///
    /// This is a `[T][L]` matrix whose element `[t][l]` presents the model
    /// expectation (marginal probability) of the state (t,l)
    /// This member is available only with CTXF_MARGINALS flag.
    mexp_state: Vec<f64>,
    /// Model expectations of transitions.
    ///
    /// This is a `[L][L]` matrix whose element `[i][j]` presents the model
    /// expectation of the transition (i--j).
    /// This member is available only with `CTXF_MARGINALS` flag.
    mexp_trans: Vec<f64>,
}

impl Context {
    pub fn new(flag: Flag, l: u32, t: u32) -> Self {
        let l = l as usize;
        let trans = vec![0.0; l * l];
        let (exp_trans, mexp_trans) = if flag.contains(Flag::MARGINALS) {
            (vec![0.0; l * l + 4], vec![0.0; l * l])
        } else {
            (Vec::new(), Vec::new())
        };
        let trans_t = vec![0.0; l * l];
        let mut ctx = Self {
            flag,
            trans,
            trans_t,
            exp_trans,
            mexp_trans,
            num_items: 0,
            num_labels: l as u32,
            ..Default::default()
        };
        ctx.set_num_items(t);
        // t gives the 'hint' for maximum length of items.
        ctx.num_items = 0;
        ctx
    }

    pub fn set_num_items(&mut self, t: u32) {
        self.num_items = t;
        if self.cap_items < t {
            let l = self.num_labels as usize;
            let t = t as usize;
            self.alpha_score = vec![0.0; t * l];
            self.beta_score = vec![0.0; t * l];
            self.scale_factor = vec![0.0; t];
            self.row = vec![0.0; l];
            if self.flag.contains(Flag::VITERBI) {
                self.backward_edge = vec![0; t * l];
            }
            self.state = vec![0.0; t * l];
            if self.flag.contains(Flag::MARGINALS) {
                self.exp_state = vec![0.0; t * l + 4];
                self.mexp_state = vec![0.0; t * l];
            }
            self.cap_items = t as u32;
        }
    }

    pub fn reset(&mut self, flag: Reset) {
        let t = self.num_items as usize;
        let l = self.num_labels as usize;
        if flag.contains(Reset::STATE) {
            self.state[..t * l].fill(0.0);
        }
        if flag.contains(Reset::TRANS) {
            self.trans[..l * l].fill(0.0);
        }
        if self.flag.contains(Flag::MARGINALS) {
            self.mexp_state[..t * l].fill(0.0);
            self.mexp_trans[..l * l].fill(0.0);
            self.log_norm = 0.0;
        }
    }

    pub fn exp_transition(&mut self) {
        let l = self.num_labels as usize;
        self.exp_trans[..l * l].copy_from_slice(&self.trans);
        for i in 0..(l * l) {
            self.exp_trans[i] = self.exp_trans[i].exp();
        }
    }

    /// Specialized Viterbi for small fixed L (fully unrolled)
    #[inline]
    fn viterbi_specialized<const L: usize>(&mut self) -> (Vec<u32>, f64) {
        // Compute the scores at (0, *)
        let current = &mut self.alpha_score;
        let state = &mut self.state;
        current[..L].clone_from_slice(&state[..L]);
        
        // Compute the scores at (t, *)
        for t in 1..self.num_items as usize {
            let (prev, current) = self.alpha_score.split_at_mut(L * t);
            let prev = &prev[L * (t - 1)..];
            let state = &self.state[L * t..];
            let back = &mut self.backward_edge[L * t..];
            
            // Compute the score of (t, j) - fully unrolled for const L
            for j in 0..L {
                let mut max_score = f64::MIN;
                let mut argmax_score = 0;
                let trans_col = &self.trans_t[L * j..];
                
                // This loop will be fully unrolled by the compiler
                for i in 0..L {
                    let score = prev[i] + trans_col[i];
                    if max_score < score {
                        max_score = score;
                        argmax_score = i;
                    }
                }
                
                back[j] = argmax_score as u32;
                current[j] = max_score + state[j];
            }
        }
        
        // Find the maximum score at the end
        let mut max_score = f64::MIN;
        let prev = &self.alpha_score[L * (self.num_items as usize - 1)..];
        let mut labels = vec![0u32; self.num_items as usize];
        
        for i in 0..L {
            if max_score < prev[i] {
                max_score = prev[i];
                labels[self.num_items as usize - 1] = i as u32;
            }
        }
        
        // Tag labels by tracing the backward links
        for t in (0..(self.num_items as usize - 1)).rev() {
            let back = &self.backward_edge[L * (t + 1)..];
            labels[t] = back[labels[t + 1] as usize];
        }
        
        (labels, max_score)
    }

    pub fn viterbi(&mut self) -> (Vec<u32>, f64) {
        let l = self.num_labels as usize;
        
        // Use specialized versions for common small L values
        // These are fully unrolled by the compiler for maximum performance
        match l {
            2 => return self.viterbi_specialized::<2>(),
            3 => return self.viterbi_specialized::<3>(),
            4 => return self.viterbi_specialized::<4>(),
            5 => return self.viterbi_specialized::<5>(),
            _ => {} // Fall through to generic version
        }
        
        // Compute the scores at (0, *)
        let current = &mut self.alpha_score;
        let state = &mut self.state;
        current[..l].clone_from_slice(&state[..l]);
        // Compute the scores at (t, *)
        for t in 1..self.num_items as usize {
            let (prev, current) = self.alpha_score.split_at_mut(l * t);
            let prev = &prev[l * (t - 1)..];
            let state = &self.state[l * t..];
            let back = &mut self.backward_edge[l * t..];
            // Compute the score of (t, j)
            for j in 0..l {
                let mut max_score = f64::MIN;
                let mut argmax_score = None;
                // Use transposed matrix for cache-friendly sequential access
                let trans_col = &self.trans_t[l * j..];
                for i in 0..l {
                    // Transit from (t-1, i) to (t, j)
                    // trans_t[j][i] = trans[i][j]
                    let score = prev[i] + trans_col[i];
                    // Store this path if it has the maximum score
                    if max_score < score {
                        max_score = score;
                        argmax_score = Some(i);
                    }
                }
                // Backward link (#t, #j) -> (#t-1, #i)
                if let Some(argmax_score) = argmax_score {
                    back[j] = argmax_score as u32;
                }
                // Add the state score on (t, j)
                current[j] = max_score + state[j];
            }
        }
        // Find the node (#T, Ei) that reaches EOS with the maximum score
        let mut max_score = f64::MIN;
        let prev = &self.alpha_score[l * (self.num_items as usize - 1)..];
        // Set a score for T-1 to be overwritten later. Just in case we don't
        // end up with something beating f64::MIN.
        let mut labels = vec![0u32; self.num_items as usize];
        for (i, prev_value) in prev.iter().enumerate().take(l) {
            if max_score < *prev_value {
                max_score = *prev_value;
                // Tag the item #T
                labels[self.num_items as usize - 1] = i as u32;
            }
        }
        // Tag labels by tracing the backward links
        for t in (0..(self.num_items as usize - 1)).rev() {
            let back = &self.backward_edge[l * (t + 1)..];
            labels[t] = back[labels[t + 1] as usize];
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
