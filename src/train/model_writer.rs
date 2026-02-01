use std::fs::File;
use std::io::{self, Seek, SeekFrom, Write};
use std::path::Path;

use cqdb::CQDBWriter;

use super::dictionary::Dictionary;
use super::feature_gen::{Feature, FeatureGenerator, FeatureRefs, FeatureType};

/// Pruned model data for serialization
struct PrunedModel {
    /// Pruned features (non-zero weights only)
    features: Vec<Feature>,
    /// Pruned attribute refs (remapped feature IDs)
    attr_refs: Vec<FeatureRefs>,
    /// Pruned label refs (remapped feature IDs)
    label_refs: Vec<FeatureRefs>,
    /// Pruned attribute dictionary (only attrs with surviving features)
    attrs: Dictionary,
}

impl PrunedModel {
    /// Create a pruned model from a feature generator
    fn from_fgen(fgen: &FeatureGenerator, attrs: &Dictionary, labels: &Dictionary) -> Self {
        let num_labels = labels.len();

        // Step 1: Build feature map (old_fid -> new_fid) for non-zero features
        let mut fmap: Vec<Option<u32>> = vec![None; fgen.features.len()];
        let mut pruned_features = Vec::new();

        for (old_fid, feature) in fgen.features.iter().enumerate() {
            if feature.weight != 0.0 {
                let new_fid = pruned_features.len() as u32;
                fmap[old_fid] = Some(new_fid);
                pruned_features.push(feature.clone());
            }
        }

        // Step 2: Build attribute map (old_aid -> new_aid) for attrs with surviving state features
        let mut amap: Vec<Option<u32>> = vec![None; attrs.len()];
        let mut pruned_attrs = Dictionary::new();

        for (old_aid, attr_ref) in fgen.attr_refs.iter().enumerate() {
            let has_surviving_feature = attr_ref
                .fids
                .iter()
                .any(|&fid| fmap[fid as usize].is_some());

            if has_surviving_feature {
                if let Some(name) = attrs.get_name(old_aid as u32) {
                    let new_aid = pruned_attrs.get_or_insert(name);
                    amap[old_aid] = Some(new_aid);
                }
            }
        }

        // Step 3: Remap feature src IDs for state features only
        // State features: src = attribute ID (needs remapping)
        // Transition features: src = previous label ID (no remapping needed)
        for feature in &mut pruned_features {
            if feature.ftype == FeatureType::State {
                let old_aid = feature.src as usize;
                if old_aid < amap.len() {
                    if let Some(new_aid) = amap[old_aid] {
                        feature.src = new_aid;
                    }
                }
            }
            // Transition features keep their src (prev_label ID) unchanged
        }

        // Step 4: Build pruned attr_refs with remapped feature IDs
        let mut pruned_attr_refs = vec![FeatureRefs::default(); pruned_attrs.len()];
        for (old_aid, attr_ref) in fgen.attr_refs.iter().enumerate() {
            if let Some(new_aid) = amap[old_aid] {
                let new_ref = &mut pruned_attr_refs[new_aid as usize];
                for &old_fid in &attr_ref.fids {
                    if let Some(new_fid) = fmap[old_fid as usize] {
                        new_ref.fids.push(new_fid);
                    }
                }
            }
        }

        // Step 5: Build pruned label_refs with remapped feature IDs
        let mut pruned_label_refs = vec![FeatureRefs::default(); num_labels];
        for (label_id, label_ref) in fgen.label_refs.iter().enumerate() {
            if label_id < num_labels {
                let new_ref = &mut pruned_label_refs[label_id];
                for &old_fid in &label_ref.fids {
                    if let Some(new_fid) = fmap[old_fid as usize] {
                        new_ref.fids.push(new_fid);
                    }
                }
            }
        }

        Self {
            features: pruned_features,
            attr_refs: pruned_attr_refs,
            label_refs: pruned_label_refs,
            attrs: pruned_attrs,
        }
    }

    fn num_features(&self) -> usize {
        self.features.len()
    }
}

/// Write a trained CRF model to file
pub struct ModelWriter;

impl ModelWriter {
    /// Write model to file in CRFsuite format
    ///
    /// This method prunes zero-weight features and unused attributes before
    /// writing, resulting in smaller model files. This matches CRFsuite's
    /// default behavior.
    pub fn write(
        filename: &Path,
        fgen: &FeatureGenerator,
        labels: &Dictionary,
        attrs: &Dictionary,
    ) -> io::Result<()> {
        // Prune zero-weight features and unused attributes
        let pruned = PrunedModel::from_fgen(fgen, attrs, labels);

        let mut file = File::create(filename)?;

        // Helper to convert stream position to u32 with overflow check
        let pos_to_u32 = |pos: u64| -> io::Result<u32> {
            u32::try_from(pos).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "file position exceeds u32::MAX")
            })
        };

        // Write header
        Self::write_header_pruned(&mut file, &pruned, labels)?;

        // Write features
        let off_features = pos_to_u32(file.stream_position()?)?;
        Self::write_features_pruned(&mut file, &pruned)?;

        // Write label dictionary (CQDB)
        let off_labels = pos_to_u32(file.stream_position()?)?;
        Self::write_cqdb(&mut file, labels)?;

        // Write attribute dictionary (CQDB) - use pruned attrs
        let off_attrs = pos_to_u32(file.stream_position()?)?;
        Self::write_cqdb(&mut file, &pruned.attrs)?;

        // Write label feature references
        Self::align_to_u32(&mut file)?;
        let off_label_refs = pos_to_u32(file.stream_position()?)?;
        Self::write_label_refs_pruned(&mut file, &pruned)?;

        // Write attribute feature references
        Self::align_to_u32(&mut file)?;
        let off_attr_refs = pos_to_u32(file.stream_position()?)?;
        Self::write_attr_refs_pruned(&mut file, &pruned)?;

        // Update header with correct offsets
        let file_size = pos_to_u32(file.stream_position()?)?;
        file.seek(SeekFrom::Start(0))?;
        Self::write_header_with_offsets_pruned(
            &mut file,
            &pruned,
            labels,
            off_features,
            off_labels,
            off_attrs,
            off_label_refs,
            off_attr_refs,
            file_size,
        )?;

        Ok(())
    }

    /// Align the file position to a 4-byte boundary with zero padding.
    fn align_to_u32(file: &mut File) -> io::Result<()> {
        let mut pos = file.stream_position()?;
        while pos % 4 != 0 {
            file.write_all(&[0])?;
            pos += 1;
        }
        Ok(())
    }

    /// Write CQDB dictionary
    fn write_cqdb(file: &mut File, dict: &Dictionary) -> io::Result<()> {
        let mut writer = CQDBWriter::new(file)?;

        // Write all dictionary entries
        for (s, id) in dict.iter() {
            writer.put(s, id)?;
        }

        // CQDBWriter automatically closes and writes the database on drop.
        // Note: If the drop implementation encounters I/O errors during flush,
        // they are silently ignored (see CQDB crate's Drop impl). This is a
        // limitation of the CQDB API which doesn't expose an explicit close()
        // method that could propagate errors.
        Ok(())
    }

    /// Write file header for pruned model
    fn write_header_pruned(
        file: &mut File,
        pruned: &PrunedModel,
        labels: &Dictionary,
    ) -> io::Result<()> {
        file.write_all(b"lCRF")?;
        file.write_all(&0u32.to_le_bytes())?;
        file.write_all(b"FOMC")?;
        file.write_all(&100u32.to_le_bytes())?;
        file.write_all(&(pruned.num_features() as u32).to_le_bytes())?;
        file.write_all(&(labels.len() as u32).to_le_bytes())?;
        file.write_all(&(pruned.attrs.len() as u32).to_le_bytes())?;
        file.write_all(&0u32.to_le_bytes())?;
        file.write_all(&0u32.to_le_bytes())?;
        file.write_all(&0u32.to_le_bytes())?;
        file.write_all(&0u32.to_le_bytes())?;
        file.write_all(&0u32.to_le_bytes())?;
        Ok(())
    }

    /// Write header with actual offsets for pruned model
    #[allow(clippy::too_many_arguments)]
    fn write_header_with_offsets_pruned(
        file: &mut File,
        pruned: &PrunedModel,
        labels: &Dictionary,
        off_features: u32,
        off_labels: u32,
        off_attrs: u32,
        off_label_refs: u32,
        off_attr_refs: u32,
        file_size: u32,
    ) -> io::Result<()> {
        file.write_all(b"lCRF")?;
        file.write_all(&file_size.to_le_bytes())?;
        file.write_all(b"FOMC")?;
        file.write_all(&100u32.to_le_bytes())?;
        file.write_all(&(pruned.num_features() as u32).to_le_bytes())?;
        file.write_all(&(labels.len() as u32).to_le_bytes())?;
        file.write_all(&(pruned.attrs.len() as u32).to_le_bytes())?;
        file.write_all(&off_features.to_le_bytes())?;
        file.write_all(&off_labels.to_le_bytes())?;
        file.write_all(&off_attrs.to_le_bytes())?;
        file.write_all(&off_label_refs.to_le_bytes())?;
        file.write_all(&off_attr_refs.to_le_bytes())?;
        Ok(())
    }

    /// Write features section for pruned model
    fn write_features_pruned(file: &mut File, pruned: &PrunedModel) -> io::Result<()> {
        file.write_all(b"FEAT")?;

        let num_features_u32 = u32::try_from(pruned.num_features()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "number of features does not fit into u32",
            )
        })?;
        let chunk_size_u64 = 12u64 + (num_features_u32 as u64) * 20u64;
        let chunk_size_u32 = u32::try_from(chunk_size_u64).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "feature chunk size exceeds u32::MAX",
            )
        })?;
        file.write_all(&chunk_size_u32.to_le_bytes())?;
        file.write_all(&num_features_u32.to_le_bytes())?;

        for feature in &pruned.features {
            let ftype = feature.ftype as u32;
            file.write_all(&ftype.to_le_bytes())?;
            file.write_all(&feature.src.to_le_bytes())?;
            file.write_all(&feature.dst.to_le_bytes())?;
            file.write_all(&feature.weight.to_le_bytes())?;
        }

        Ok(())
    }

    /// Write label feature references for pruned model
    fn write_label_refs_pruned(file: &mut File, pruned: &PrunedModel) -> io::Result<()> {
        let num_labels = pruned.label_refs.len();
        let total_labels = num_labels
            .checked_add(2)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "label count overflow"))?;
        let chunk_start = u32::try_from(file.stream_position()?).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "chunk start position exceeds u32::MAX",
            )
        })?;

        file.write_all(b"LFRF")?;
        let num_labels_u32 = u32::try_from(total_labels).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "number of labels exceeds u32::MAX",
            )
        })?;
        let header_size_u64 = 12u64 + (num_labels_u32 as u64) * 4u64;
        let header_size_u32 = u32::try_from(header_size_u64).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "label refs header size exceeds u32::MAX",
            )
        })?;
        file.write_all(&0u32.to_le_bytes())?;
        file.write_all(&num_labels_u32.to_le_bytes())?;

        let mut current_offset = chunk_start + header_size_u32;
        let mut offsets = vec![0u32; total_labels];

        for (index, label_ref) in pruned.label_refs.iter().enumerate() {
            offsets[index] = current_offset;
            let fids_len_u32 = u32::try_from(label_ref.fids.len()).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "feature count for label exceeds u32::MAX",
                )
            })?;
            current_offset = current_offset
                .checked_add(
                    4u32.checked_add(fids_len_u32.checked_mul(4).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "offset calculation overflow")
                    })?)
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "offset calculation overflow")
                    })?,
                )
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "offset overflow"))?;
        }

        for offset in &offsets {
            file.write_all(&offset.to_le_bytes())?;
        }

        for label_ref in &pruned.label_refs {
            file.write_all(&(label_ref.fids.len() as u32).to_le_bytes())?;
            for &fid in &label_ref.fids {
                file.write_all(&fid.to_le_bytes())?;
            }
        }

        let end_pos = file.stream_position()?;
        let chunk_size_u64 = end_pos
            .checked_sub(u64::from(chunk_start))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "chunk size underflow"))?;
        let chunk_size_u32 = u32::try_from(chunk_size_u64).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "label refs chunk size exceeds u32::MAX",
            )
        })?;
        file.seek(SeekFrom::Start(u64::from(chunk_start) + 4))?;
        file.write_all(&chunk_size_u32.to_le_bytes())?;
        file.seek(SeekFrom::Start(end_pos))?;

        Ok(())
    }

    /// Write attribute feature references for pruned model
    fn write_attr_refs_pruned(file: &mut File, pruned: &PrunedModel) -> io::Result<()> {
        let num_attrs = pruned.attr_refs.len();
        let chunk_start = u32::try_from(file.stream_position()?).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "chunk start position exceeds u32::MAX",
            )
        })?;

        file.write_all(b"AFRF")?;
        let num_attrs_u32 = u32::try_from(num_attrs).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "number of attrs exceeds u32::MAX",
            )
        })?;
        let header_size_u64 = 12u64 + (num_attrs_u32 as u64) * 4u64;
        let header_size_u32 = u32::try_from(header_size_u64).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "attr refs header size exceeds u32::MAX",
            )
        })?;
        file.write_all(&0u32.to_le_bytes())?;
        file.write_all(&num_attrs_u32.to_le_bytes())?;

        let mut current_offset = chunk_start + header_size_u32;
        let mut offsets = Vec::new();

        for attr_ref in &pruned.attr_refs {
            offsets.push(current_offset);
            let fids_len_u32 = u32::try_from(attr_ref.fids.len()).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "feature count for attr exceeds u32::MAX",
                )
            })?;
            current_offset = current_offset
                .checked_add(
                    4u32.checked_add(fids_len_u32.checked_mul(4).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "offset calculation overflow")
                    })?)
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "offset calculation overflow")
                    })?,
                )
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "offset overflow"))?;
        }

        for offset in &offsets {
            file.write_all(&offset.to_le_bytes())?;
        }

        for attr_ref in &pruned.attr_refs {
            file.write_all(&(attr_ref.fids.len() as u32).to_le_bytes())?;
            for &fid in &attr_ref.fids {
                file.write_all(&fid.to_le_bytes())?;
            }
        }

        let end_pos = file.stream_position()?;
        let chunk_size_u64 = end_pos
            .checked_sub(u64::from(chunk_start))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "chunk size underflow"))?;
        let chunk_size_u32 = u32::try_from(chunk_size_u64).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "attr refs chunk size exceeds u32::MAX",
            )
        })?;
        file.seek(SeekFrom::Start(u64::from(chunk_start) + 4))?;
        file.write_all(&chunk_size_u32.to_le_bytes())?;
        file.seek(SeekFrom::Start(end_pos))?;

        Ok(())
    }
}
