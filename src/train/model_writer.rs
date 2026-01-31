use std::fs::File;
use std::io::{self, Seek, SeekFrom, Write};
use std::path::Path;

use cqdb::CQDBWriter;

use super::dictionary::Dictionary;
use super::feature_gen::FeatureGenerator;

/// Write a trained CRF model to file
pub struct ModelWriter;

impl ModelWriter {
    /// Write model to file in CRFsuite format
    pub fn write(
        filename: &Path,
        fgen: &FeatureGenerator,
        labels: &Dictionary,
        attrs: &Dictionary,
    ) -> io::Result<()> {
        let mut file = File::create(filename)?;

        // Helper to convert stream position to u32 with overflow check
        let pos_to_u32 = |pos: u64| -> io::Result<u32> {
            u32::try_from(pos).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "file position exceeds u32::MAX")
            })
        };

        // Write header
        Self::write_header(&mut file, fgen, labels, attrs)?;

        // Write features
        let off_features = pos_to_u32(file.stream_position()?)?;
        Self::write_features(&mut file, fgen)?;

        // Write label dictionary (CQDB)
        let off_labels = pos_to_u32(file.stream_position()?)?;
        Self::write_cqdb(&mut file, labels)?;

        // Write attribute dictionary (CQDB)
        let off_attrs = pos_to_u32(file.stream_position()?)?;
        Self::write_cqdb(&mut file, attrs)?;

        // Write label feature references
        Self::align_to_u32(&mut file)?;
        let off_label_refs = pos_to_u32(file.stream_position()?)?;
        Self::write_label_refs(&mut file, fgen)?;

        // Write attribute feature references
        Self::align_to_u32(&mut file)?;
        let off_attr_refs = pos_to_u32(file.stream_position()?)?;
        Self::write_attr_refs(&mut file, fgen)?;

        // Update header with correct offsets
        let file_size = pos_to_u32(file.stream_position()?)?;
        file.seek(SeekFrom::Start(0))?;
        Self::write_header_with_offsets(
            &mut file,
            fgen,
            labels,
            attrs,
            off_features,
            off_labels,
            off_attrs,
            off_label_refs,
            off_attr_refs,
            file_size,
        )?;

        Ok(())
    }

    /// Write file header
    fn write_header(
        file: &mut File,
        fgen: &FeatureGenerator,
        labels: &Dictionary,
        attrs: &Dictionary,
    ) -> io::Result<()> {
        // Write placeholder header (will be updated later)
        file.write_all(b"lCRF")?; // magic
        file.write_all(&0u32.to_le_bytes())?; // size (placeholder)
        file.write_all(b"FOMC")?; // type
        file.write_all(&100u32.to_le_bytes())?; // version
        file.write_all(&(fgen.num_features() as u32).to_le_bytes())?;
        file.write_all(&(labels.len() as u32).to_le_bytes())?;
        file.write_all(&(attrs.len() as u32).to_le_bytes())?;
        file.write_all(&0u32.to_le_bytes())?; // off_features (placeholder)
        file.write_all(&0u32.to_le_bytes())?; // off_labels (placeholder)
        file.write_all(&0u32.to_le_bytes())?; // off_attrs (placeholder)
        file.write_all(&0u32.to_le_bytes())?; // off_label_refs (placeholder)
        file.write_all(&0u32.to_le_bytes())?; // off_attr_refs (placeholder)
        Ok(())
    }

    /// Write header with actual offsets
    #[allow(clippy::too_many_arguments)]
    fn write_header_with_offsets(
        file: &mut File,
        fgen: &FeatureGenerator,
        labels: &Dictionary,
        attrs: &Dictionary,
        off_features: u32,
        off_labels: u32,
        off_attrs: u32,
        off_label_refs: u32,
        off_attr_refs: u32,
        file_size: u32,
    ) -> io::Result<()> {
        file.write_all(b"lCRF")?; // magic
        file.write_all(&file_size.to_le_bytes())?; // size
        file.write_all(b"FOMC")?; // type
        file.write_all(&100u32.to_le_bytes())?; // version
        file.write_all(&(fgen.num_features() as u32).to_le_bytes())?;
        file.write_all(&(labels.len() as u32).to_le_bytes())?;
        file.write_all(&(attrs.len() as u32).to_le_bytes())?;
        file.write_all(&off_features.to_le_bytes())?;
        file.write_all(&off_labels.to_le_bytes())?;
        file.write_all(&off_attrs.to_le_bytes())?;
        file.write_all(&off_label_refs.to_le_bytes())?;
        file.write_all(&off_attr_refs.to_le_bytes())?;
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

    /// Write features section
    fn write_features(file: &mut File, fgen: &FeatureGenerator) -> io::Result<()> {
        // Write chunk header
        file.write_all(b"FEAT")?; // chunk ID

        // Use checked arithmetic to detect overflow
        let num_features_u32 = u32::try_from(fgen.num_features()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "number of features does not fit into u32",
            )
        })?;
        let chunk_size_u64 = 12u64 + (num_features_u32 as u64) * 20u64; // header + features
        let chunk_size_u32 = u32::try_from(chunk_size_u64).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "feature chunk size exceeds u32::MAX",
            )
        })?;
        file.write_all(&chunk_size_u32.to_le_bytes())?;
        file.write_all(&num_features_u32.to_le_bytes())?;

        // Write each feature
        // Feature weight is stored as a 64-bit IEEE 754 float in little-endian order,
        // as required by the CRFsuite binary model format.
        for feature in &fgen.features {
            let ftype = feature.ftype as u32;
            file.write_all(&ftype.to_le_bytes())?;
            file.write_all(&feature.src.to_le_bytes())?;
            file.write_all(&feature.dst.to_le_bytes())?;
            file.write_all(&feature.weight.to_le_bytes())?;
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

    /// Write label feature references
    fn write_label_refs(file: &mut File, fgen: &FeatureGenerator) -> io::Result<()> {
        let num_labels = fgen.label_refs.len();
        let total_labels = num_labels
            .checked_add(2)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "label count overflow"))?;
        let chunk_start = u32::try_from(file.stream_position()?).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "chunk start position exceeds u32::MAX",
            )
        })?;

        // Write chunk header with checked arithmetic
        file.write_all(b"LFRF")?; // chunk ID
        let num_labels_u32 = u32::try_from(total_labels).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "number of labels exceeds u32::MAX",
            )
        })?;
        let header_size_u64 = 12u64 + (num_labels_u32 as u64) * 4u64; // header + offsets
        let header_size_u32 = u32::try_from(header_size_u64).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "label refs header size exceeds u32::MAX",
            )
        })?;
        // Placeholder size; will be updated after writing all lists, matching CRFsuite.
        file.write_all(&0u32.to_le_bytes())?;
        file.write_all(&num_labels_u32.to_le_bytes())?;

        // Calculate offsets for each label's feature list (absolute offsets)
        let mut current_offset = chunk_start + header_size_u32;
        let mut offsets = vec![0u32; total_labels];

        for (index, label_ref) in fgen.label_refs.iter().enumerate() {
            offsets[index] = current_offset;
            // Use checked arithmetic for offset calculation
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

        // Write offset table
        for offset in &offsets {
            file.write_all(&offset.to_le_bytes())?;
        }

        // Write feature ID lists
        for label_ref in &fgen.label_refs {
            file.write_all(&(label_ref.fids.len() as u32).to_le_bytes())?;
            for &fid in &label_ref.fids {
                file.write_all(&fid.to_le_bytes())?;
            }
        }

        // Update chunk size to include header, offsets, and lists
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

    /// Write attribute feature references
    fn write_attr_refs(file: &mut File, fgen: &FeatureGenerator) -> io::Result<()> {
        let num_attrs = fgen.attr_refs.len();
        let chunk_start = u32::try_from(file.stream_position()?).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "chunk start position exceeds u32::MAX",
            )
        })?;

        // Write chunk header with checked arithmetic
        file.write_all(b"AFRF")?; // chunk ID
        let num_attrs_u32 = u32::try_from(num_attrs).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "number of attrs exceeds u32::MAX",
            )
        })?;
        let header_size_u64 = 12u64 + (num_attrs_u32 as u64) * 4u64; // header + offsets
        let header_size_u32 = u32::try_from(header_size_u64).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "attr refs header size exceeds u32::MAX",
            )
        })?;
        // Placeholder size; will be updated after writing all lists, matching CRFsuite.
        file.write_all(&0u32.to_le_bytes())?;
        file.write_all(&num_attrs_u32.to_le_bytes())?;

        // Calculate offsets for each attribute's feature list (absolute offsets)
        let mut current_offset = chunk_start + header_size_u32;
        let mut offsets = Vec::new();

        for attr_ref in &fgen.attr_refs {
            offsets.push(current_offset);
            // Use checked arithmetic for offset calculation
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

        // Write offset table
        for offset in &offsets {
            file.write_all(&offset.to_le_bytes())?;
        }

        // Write feature ID lists
        for attr_ref in &fgen.attr_refs {
            file.write_all(&(attr_ref.fids.len() as u32).to_le_bytes())?;
            for &fid in &attr_ref.fids {
                file.write_all(&fid.to_le_bytes())?;
            }
        }

        // Update chunk size to include header, offsets, and lists
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
