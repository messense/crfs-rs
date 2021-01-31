use std::io;

use crate::model::unpack_u32;

#[derive(Debug, Clone)]
pub struct Feature {
    pub r#type: u32,
    pub source: u32,
    pub target: u32,
    pub weight: f64,
}

/// Feature references
///
/// This is a collection of feature ids used for faster accesses.
#[derive(Debug, Clone)]
pub struct FeatureRefs<'a> {
    pub num_features: u32,
    pub feature_ids: &'a [u8],
}

impl<'a> FeatureRefs<'a> {
    pub fn get(&self, index: usize) -> io::Result<u32> {
        let fid = unpack_u32(&self.feature_ids[index * 4..])?;
        Ok(fid)
    }
}
