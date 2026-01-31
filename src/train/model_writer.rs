use std::fs::File;
use std::io::{self, Seek, SeekFrom, Write};

use cqdb::CQDBWriter;

use super::dictionary::Dictionary;
use super::feature_gen::FeatureGenerator;

/// Write a trained CRF model to file
pub struct ModelWriter;

impl ModelWriter {
    /// Write model to file in CRFsuite format
    pub fn write(
        filename: &str,
        fgen: &FeatureGenerator,
        labels: &Dictionary,
        attrs: &Dictionary,
    ) -> io::Result<()> {
        let mut file = File::create(filename)?;

        // Write header
        Self::write_header(&mut file, fgen, labels, attrs)?;

        // Write features
        let off_features = file.stream_position()? as u32;
        Self::write_features(&mut file, fgen)?;

        // Write label dictionary (CQDB)
        let off_labels = file.stream_position()? as u32;
        Self::write_cqdb(&mut file, labels)?;

        // Write attribute dictionary (CQDB)
        let off_attrs = file.stream_position()? as u32;
        Self::write_cqdb(&mut file, attrs)?;

        // Write label feature references
        let off_label_refs = file.stream_position()? as u32;
        Self::write_label_refs(&mut file, fgen)?;

        // Write attribute feature references
        let off_attr_refs = file.stream_position()? as u32;
        Self::write_attr_refs(&mut file, fgen)?;

        // Update header with correct offsets
        let file_size = file.stream_position()? as u32;
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

    /// Write features section
    fn write_features(file: &mut File, fgen: &FeatureGenerator) -> io::Result<()> {
        // Write chunk header
        file.write_all(b"FEAT")?; // chunk ID
        let chunk_size = 12 + fgen.num_features() * 20; // header + features
        file.write_all(&(chunk_size as u32).to_le_bytes())?;
        file.write_all(&(fgen.num_features() as u32).to_le_bytes())?;

        // Write each feature
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

        // CQDBWriter will automatically close and write the database on drop
        Ok(())
    }

    /// Write label feature references
    fn write_label_refs(file: &mut File, fgen: &FeatureGenerator) -> io::Result<()> {
        let num_labels = fgen.label_refs.len();
        let chunk_start = file.stream_position()? as u32;

        // Write chunk header
        file.write_all(b"LREF")?; // chunk ID
        let chunk_size = 12 + num_labels * 4; // header + offsets
        file.write_all(&(chunk_size as u32).to_le_bytes())?;
        file.write_all(&(num_labels as u32).to_le_bytes())?;

        // Calculate offsets for each label's feature list (absolute offsets)
        let mut current_offset = chunk_start + chunk_size as u32;
        let mut offsets = Vec::new();

        for label_ref in &fgen.label_refs {
            offsets.push(current_offset);
            current_offset += 4 + label_ref.fids.len() as u32 * 4; // count + fids
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

        Ok(())
    }

    /// Write attribute feature references
    fn write_attr_refs(file: &mut File, fgen: &FeatureGenerator) -> io::Result<()> {
        let num_attrs = fgen.attr_refs.len();
        let chunk_start = file.stream_position()? as u32;

        // Write chunk header
        file.write_all(b"AREF")?; // chunk ID
        let chunk_size = 12 + num_attrs * 4; // header + offsets
        file.write_all(&(chunk_size as u32).to_le_bytes())?;
        file.write_all(&(num_attrs as u32).to_le_bytes())?;

        // Calculate offsets for each attribute's feature list (absolute offsets)
        let mut current_offset = chunk_start + chunk_size as u32;
        let mut offsets = Vec::new();

        for attr_ref in &fgen.attr_refs {
            offsets.push(current_offset);
            current_offset += 4 + attr_ref.fids.len() as u32 * 4; // count + fids
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

        Ok(())
    }
}
