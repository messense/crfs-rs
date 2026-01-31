use std::{
    convert::TryInto,
    fmt,
    io::{self, Write},
    mem,
};

use bstr::ByteSlice;
use cqdb::CQDB;

use crate::Tagger;
use crate::feature::{Feature, FeatureRefs};

const CHUNK_SIZE: usize = 12;
const FEATURE_SIZE: usize = 20;

#[inline]
pub(crate) fn unpack_u32(buf: &[u8]) -> io::Result<u32> {
    if buf.len() < 4 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "not enough data for unpacking u32",
        ));
    }
    Ok(u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]))
}

#[inline]
fn unpack_f64(buf: &[u8]) -> io::Result<f64> {
    if buf.len() < 8 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "not enough data for unpacking f64",
        ));
    }
    Ok(f64::from_le_bytes([
        buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
    ]))
}

#[derive(Debug, Clone)]
#[repr(C)]
struct Header {
    magic: [u8; 4],
    size: u32,
    r#type: [u8; 4],
    version: u32,
    num_features: u32,
    num_labels: u32,
    num_attrs: u32,
    off_features: u32,
    off_labels: u32,
    off_attrs: u32,
    off_label_refs: u32,
    off_attr_refs: u32,
}

/// The CRF model
#[derive(Clone)]
pub struct Model<'a> {
    buffer: &'a [u8],
    size: u32,
    header: Header,
    labels: CQDB<'a>,
    attrs: CQDB<'a>,
}

impl<'a> fmt::Debug for Model<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Model")
            .field("size", &self.size)
            .field("header", &self.header)
            .field("labels", &self.labels)
            .field("attrs", &self.attrs)
            .finish()
    }
}

impl<'a> Model<'a> {
    /// Create an instance of a model object from a model in memory
    pub fn new(buf: &'a [u8]) -> io::Result<Self> {
        let size = buf.len();
        if size <= mem::size_of::<Header>() {
            return Err(io::Error::other("invalid model format"));
        }
        let magic = &buf[0..4];
        if magic != b"lCRF" {
            return Err(io::Error::other("invalid file format, magic mismatch"));
        }
        let mut index = 4;
        let header_size = unpack_u32(&buf[index..])?;
        index += 4;
        let header_type = &buf[index..index + 4];
        index += 4;
        let version = unpack_u32(&buf[index..])?;
        index += 4;
        let num_features = unpack_u32(&buf[index..])?;
        index += 4;
        let num_labels = unpack_u32(&buf[index..])?;
        index += 4;
        let num_attrs = unpack_u32(&buf[index..])?;
        index += 4;
        let off_features = unpack_u32(&buf[index..])?;
        index += 4;
        let off_labels = unpack_u32(&buf[index..])?;
        index += 4;
        let off_attrs = unpack_u32(&buf[index..])?;
        index += 4;
        let off_label_refs = unpack_u32(&buf[index..])?;
        index += 4;
        let off_attr_refs = unpack_u32(&buf[index..])?;
        let header = Header {
            magic: magic.try_into().unwrap(),
            size: header_size,
            r#type: header_type.try_into().unwrap(),
            version,
            num_features,
            num_labels,
            num_attrs,
            off_features,
            off_labels,
            off_attrs,
            off_label_refs,
            off_attr_refs,
        };
        let labels_start = off_labels as usize;
        let labels = CQDB::new(&buf[labels_start..size])?;
        let attrs_start = off_attrs as usize;
        let attrs = CQDB::new(&buf[attrs_start..size])?;
        Ok(Self {
            buffer: buf,
            size: size as u32,
            header,
            labels,
            attrs,
        })
    }

    /// Number of attributes
    pub fn num_attrs(&self) -> u32 {
        self.header.num_attrs
    }

    /// Number of labels
    pub fn num_labels(&self) -> u32 {
        self.header.num_labels
    }

    /// Convert a label ID to label string
    pub fn to_label(&self, lid: u32) -> Option<&str> {
        self.labels.to_str(lid).and_then(|s| s.to_str().ok())
    }

    /// Convert a label string to label ID
    pub fn to_label_id(&self, value: &str) -> Option<u32> {
        self.labels.to_id(value)
    }

    /// Convert a attribute ID to attribute string
    pub fn to_attr(&self, aid: u32) -> Option<&str> {
        self.attrs.to_str(aid).and_then(|s| s.to_str().ok())
    }

    /// Convert a attribute string to attribute ID
    pub fn to_attr_id(&self, value: &str) -> Option<u32> {
        self.attrs.to_id(value)
    }

    pub(crate) fn label_ref(&self, lid: u32) -> io::Result<FeatureRefs<'_>> {
        let mut index = self.header.off_label_refs as usize + CHUNK_SIZE;
        index += 4 * lid as usize;
        let offset = unpack_u32(&self.buffer[index..])? as usize;
        let num_features = unpack_u32(&self.buffer[offset..])?;
        let feature_ids = &self.buffer[offset + 4..];
        Ok(FeatureRefs {
            num_features,
            feature_ids,
        })
    }

    pub(crate) fn attr_ref(&self, lid: u32) -> io::Result<FeatureRefs<'_>> {
        let mut index = self.header.off_attr_refs as usize + CHUNK_SIZE;
        index += 4 * lid as usize;
        let offset = unpack_u32(&self.buffer[index..])? as usize;
        let num_features = unpack_u32(&self.buffer[offset..])?;
        let feature_ids = &self.buffer[offset + 4..];
        Ok(FeatureRefs {
            num_features,
            feature_ids,
        })
    }

    pub(crate) fn feature(&self, fid: u32) -> io::Result<Feature> {
        let mut index = self.header.off_features as usize + CHUNK_SIZE;
        index += FEATURE_SIZE * fid as usize;
        let r#type = unpack_u32(&self.buffer[index..])?;
        index += 4;
        let source = unpack_u32(&self.buffer[index..])?;
        index += 4;
        let target = unpack_u32(&self.buffer[index..])?;
        index += 4;
        let weight = unpack_f64(&self.buffer[index..])?;
        Ok(Feature {
            r#type,
            source,
            target,
            weight,
        })
    }

    /// Get a new tagger
    pub fn tagger(&'a self) -> io::Result<Tagger<'a>> {
        Tagger::new(self)
    }

    /// Print the model in human-readable format
    pub fn dump<W: Write>(&self, w: &mut W) -> io::Result<()> {
        // Dump the file header
        writeln!(w, "FILEHEADER = {{")?;
        let header = &self.header;
        writeln!(
            w,
            "  magic: {}",
            std::str::from_utf8(&header.magic).unwrap()
        )?;
        writeln!(w, "  size: {}", header.size)?;
        writeln!(
            w,
            "  type: {}",
            std::str::from_utf8(&header.r#type).unwrap()
        )?;
        writeln!(w, "  version: {}", header.version)?;
        writeln!(w, "  num_features: {}", header.num_features)?;
        writeln!(w, "  num_labels: {}", header.num_labels)?;
        writeln!(w, "  num_attrs: {}", header.num_attrs)?;
        writeln!(w, "  off_features: {:#X}", header.off_features)?;
        writeln!(w, "  off_labels: {:#X}", header.off_labels)?;
        writeln!(w, "  off_attrs: {:#X}", header.off_attrs)?;
        writeln!(w, "  off_labelrefs: {:#X}", header.off_label_refs)?;
        writeln!(w, "  off_attrrefs: {:#X}", header.off_attr_refs)?;
        writeln!(w, "}}\n")?;
        // Dump the labels
        writeln!(w, "LABELS = {{")?;
        for i in 0..header.num_labels {
            let label = self.to_label(i).unwrap();
            writeln!(w, "  {:>5}: {}", i, label)?;
        }
        writeln!(w, "}}\n")?;
        // Dump the attributes
        writeln!(w, "ATTRIBUTES = {{")?;
        for i in 0..header.num_attrs {
            let attr = self.to_attr(i).unwrap();
            writeln!(w, "  {:>5}: {}", i, attr)?;
        }
        writeln!(w, "}}\n")?;
        // Dump the transition features
        writeln!(w, "TRANSITIONS = {{")?;
        for i in 0..header.num_labels {
            let label_refs = self.label_ref(i)?;
            for j in 0..label_refs.num_features {
                let fid = label_refs.get(j as usize)?;
                let feature = self.feature(fid)?;
                let source = self.to_label(feature.source).unwrap();
                let target = self.to_label(feature.target).unwrap();
                writeln!(
                    w,
                    "  ({}) {} --> {}: {:.6}",
                    feature.r#type, source, target, feature.weight
                )?;
            }
        }
        writeln!(w, "}}\n")?;
        // Dump the state transition features
        writeln!(w, "STATE_FEATURES = {{")?;
        for i in 0..header.num_attrs {
            let attr_refs = self.attr_ref(i)?;
            for j in 0..attr_refs.num_features {
                let fid = attr_refs.get(j as usize)?;
                let feature = self.feature(fid)?;
                let attr = self.to_attr(feature.source).unwrap();
                let target = self.to_label(feature.target).unwrap();
                writeln!(
                    w,
                    "  ({}) {} --> {}: {:.6}",
                    feature.r#type, attr, target, feature.weight
                )?;
            }
        }
        writeln!(w, "}}\n")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Model;
    use std::fs;

    #[test]
    fn test_model_new() {
        let buf = fs::read("tests/model.crfsuite").unwrap();
        let model = Model::new(&buf).unwrap();
        assert_eq!(100, model.header.version);
        assert_eq!(0, model.header.num_features);
        assert_eq!(2, model.header.num_labels);
        assert_eq!(3, model.header.num_attrs);

        let _debug = format!("{:?}", model);
    }

    #[test]
    fn test_invalid_model() {
        let buf = b"";
        let model = Model::new(buf);
        assert!(model.is_err());

        let mut buf = fs::read("tests/model.crfsuite").unwrap();
        let offset = std::mem::size_of::<super::Header>();
        let buf = &mut buf[..offset + 10];
        buf[0] = b'L'; // change magic from lCRF to LCRF
        let model = Model::new(buf);
        assert!(model.is_err());
    }

    #[test]
    fn test_model_dump() {
        let buf = fs::read("tests/model.crfsuite").unwrap();
        let model = Model::new(&buf).unwrap();
        let mut out = Vec::new();
        model.dump(&mut out).unwrap();
        let out_str = std::str::from_utf8(&out).unwrap();
        let expected = r#"FILEHEADER = {
  magic: lCRF
  size: 4684
  type: FOMC
  version: 100
  num_features: 0
  num_labels: 2
  num_attrs: 3
  off_features: 0x30
  off_labels: 0x104
  off_attrs: 0x960
  off_labelrefs: 0x11DC
  off_attrrefs: 0x1210
}

LABELS = {
      0: sunny
      1: rainy
}

ATTRIBUTES = {
      0: walk
      1: shop
      2: clean
}

TRANSITIONS = {
  (1) sunny --> sunny: 0.200033
  (1) sunny --> rainy: 0.008212
  (1) rainy --> sunny: -0.239633
  (1) rainy --> rainy: 0.031389
}

STATE_FEATURES = {
  (0) walk --> sunny: 0.443627
  (0) walk --> rainy: -0.443627
  (0) shop --> sunny: 0.003924
  (0) shop --> rainy: -0.003924
  (0) clean --> sunny: -0.500569
  (0) clean --> rainy: 0.500569
}

"#;
        assert_eq!(out_str, expected);
    }

    #[test]
    fn test_model_tag() {
        use crate::Attribute;

        let buf = fs::read("tests/model.crfsuite").unwrap();
        let model = Model::new(&buf).unwrap();
        let tagger = model.tagger().unwrap();
        let xseq = vec![
            vec![Attribute::new("walk", 1.0), Attribute::new("shop", 0.5)],
            vec![Attribute::new("walk", 1.0)],
            vec![Attribute::new("walk", 1.0), Attribute::new("clean", 0.5)],
            vec![Attribute::new("shop", 0.5), Attribute::new("clean", 0.5)],
            vec![Attribute::new("walk", 0.5), Attribute::new("clean", 1.0)],
            vec![Attribute::new("clean", 1.0), Attribute::new("shop", 0.1)],
            vec![Attribute::new("walk", 1.0), Attribute::new("shop", 0.5)],
            vec![],
            vec![Attribute::new("clean", 1.0)],
        ];
        let yseq = [
            "sunny", "sunny", "sunny", "rainy", "rainy", "rainy", "sunny", "sunny", "rainy",
        ];
        let res = tagger.tag(&xseq).unwrap();
        assert_eq!(res, yseq);

        let tagger = model.tagger().unwrap();
        let res = tagger.tag::<[Attribute; 0]>(&[]).unwrap();
        assert!(res.is_empty());
    }
}
