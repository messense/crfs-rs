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
    pub log_norm: f64,
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
        let mut ctx = Self {
            flag,
            trans,
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
            // FIXME: use slice::fill when it reaches stable
            for el in &mut self.state[..t * l] {
                *el = 0.0;
            }
        }
        if flag.contains(Reset::TRANS) {
            for el in &mut self.trans[..l * l] {
                *el = 0.0;
            }
        }
        if self.flag.contains(Flag::MARGINALS) {
            for el in &mut self.mexp_state[..t * l] {
                *el = 0.0;
            }
            for el in &mut self.mexp_trans[..l * l] {
                *el = 0.0;
            }
            self.log_norm = 0.0;
        }
    }

    pub fn exp_state(&mut self) {
        let l = self.num_labels as usize;
        let t = self.num_items as usize;
        self.exp_state[..l * t].copy_from_slice(&self.state);
        for i in 0..(l * t) {
            self.exp_state[i] = self.exp_state[i].exp();
        }
    }

    pub fn exp_transition(&mut self) {
        let l = self.num_labels as usize;
        self.exp_trans[..l * l].copy_from_slice(&self.trans);
        for i in 0..(l * l) {
            self.exp_trans[i] = self.exp_trans[i].exp();
        }
    }

    pub fn alpha_score(&mut self) {
        let l = self.num_labels as usize;
        let scale = &mut self.scale_factor[0];
        // Compute the alpha scores on nodes (0, *)
        let current = &mut self.alpha_score;
        let state = &self.exp_state;
        current[..l].clone_from_slice(&state[..l]);
        let mut sum: f64 = current.iter().take(l).sum();
        *scale = if sum != 0.0 { 1.0 / sum } else { 1.0 };
        for i in 0..l {
            current[i] *= *scale;
        }
        *scale += 1.0;
        // Compute the alpha scores on nodes (t, *)
        for t in 1..self.num_items as usize {
            let (prev, current) = self.alpha_score.split_at_mut(l * t);
            let prev = &prev[l * (t - 1)..];
            let state = &self.exp_state[l * t..];
            for el in &mut current[..l] {
                *el = 0.0;
            }
            for j in 0..l {
                let trans = &self.exp_trans[l * j..];
                for i in 0..l {
                    current[i] += prev[j] * trans[i];
                }
            }
            for i in 0..l {
                current[i] *= state[i];
            }
            sum = current.iter().take(l).sum();
            *scale = if sum != 0.0 { 1.0 / sum } else { 1.0 };
            for i in 0..l {
                current[i] *= *scale;
            }
            *scale += 1.0;
        }
        // Compute the logarithm of the normalization factor
        self.log_norm = -self
            .scale_factor
            .iter()
            .take(self.num_items as usize)
            .map(|s| s.ln())
            .sum::<f64>();
    }

    pub fn beta_score(&mut self) {
        let l = self.num_labels as usize;
        let scale = &mut self.scale_factor[self.num_items as usize - 1];
        let row = &mut self.row;
        // Compute the beta scales at (T-1, *)
        let current = &mut self.beta_score[l * (self.num_items as usize - 1)..];
        for i in 0..l {
            current[i] = *scale;
        }
        *scale -= 1.0;
        // Compute the beta scores at (t, *)
        for t in (0..(self.num_items as usize - 1)).rev() {
            let (current, next) = self.alpha_score.split_at_mut(l * (t + 1));
            let current = &mut current[l * t..];
            let state = &self.exp_state[l * (t + 1)..];
            row[..l].clone_from_slice(next);
            for i in 0..l {
                row[i] *= state[i];
            }
            // Compute the beta score at (t, i)
            for i in 0..l {
                let trans = &self.exp_trans[l * i..];
                // dot product
                let mut val = 0.0;
                for j in 0..l {
                    val += trans[j] * row[j];
                }
                current[i] = val;
            }
            for i in 0..l {
                current[i] *= *scale;
            }
            *scale -= 1.0;
        }
    }

    pub fn score(&self, labels: &[u32]) -> f64 {
        let l = self.num_labels as usize;
        // Stay at (0, labels[0])
        let mut i = labels[0] as usize;
        let state = &self.state;
        let mut score = state[i];
        // Loop over the rest of items
        for t in 0..self.num_items as usize {
            let j = labels[t] as usize;
            let trans = &self.trans[l * i..];
            let state = &self.state[l * t..];
            // Transit from (t-1, i) to (t, j)
            score += trans[j];
            score += state[j];
            i = j;
        }
        score
    }

    pub fn marginal_point(&self, l: usize, t: usize) -> f64 {
        let num = self.num_labels as usize;
        let fwd = &self.alpha_score[num * t..];
        let bwd = &self.beta_score[num * t..];
        fwd[l] * bwd[l] / self.scale_factor[t]
    }

    pub fn marginal_path(&self, path: &[usize], begin: usize, end: usize) -> f64 {
        let l = self.num_labels as usize;
        let fwd = &self.alpha_score[l * begin..];
        let bwd = &self.beta_score[l * (end - 1)..];
        let mut prob = fwd[path[begin]] * bwd[path[end - 1]] / self.scale_factor[begin];
        for t in begin..end - 1 {
            let state = &self.exp_state[l * (t + 1)..];
            let edge = &self.trans[l * path[t]..];
            prob *= edge[path[t + 1]] * state[path[t + 1]] * self.scale_factor[t];
        }
        prob
    }

    pub fn viterbi(&mut self) -> (Vec<u32>, f64) {
        let mut score;
        let l = self.num_labels as usize;
        // Compute the scores at (0, *)
        let current = &mut self.alpha_score;
        let state = &self.state;
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
                for (i, prev_value) in prev.iter().enumerate().take(l) {
                    // Transit from (t-1, i) to (t, j)
                    let trans = &self.trans[l * i..];
                    score = prev_value + trans[j];
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
        // Tag labels by tracing teh backward links
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
