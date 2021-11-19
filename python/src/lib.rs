use std::fs;
use std::io::{self, Write};

use crfs_rs::{Attribute, Model};
use ouroboros::self_referencing;
use pyo3::prelude::*;

#[pyclass(module = "crfs", name = "Attribute")]
#[derive(FromPyObject)]
struct PyAttribute {
    /// Attribute name
    #[pyo3(get, set)]
    name: String,
    /// Value of the attribute
    #[pyo3(get, set)]
    value: f64,
}

#[pymethods]
impl PyAttribute {
    #[new]
    #[args(name, value = "1.0")]
    fn new(name: String, value: f64) -> Self {
        Self { name, value }
    }
}

#[derive(FromPyObject)]
enum PyAttributeInput {
    #[pyo3(transparent)]
    Attr(PyAttribute),
    Dict {
        /// Attribute name
        #[pyo3(item("name"))]
        name: String,
        /// Value of the attribute
        #[pyo3(item("value"))]
        value: f64,
    },
    Tuple(String, f64),
    #[pyo3(transparent)]
    NameOnly(String),
}

impl From<PyAttributeInput> for Attribute {
    fn from(attr: PyAttributeInput) -> Self {
        match attr {
            PyAttributeInput::Attr(PyAttribute { name, value }) => Attribute::new(name, value),
            PyAttributeInput::Dict { name, value } => Attribute::new(name, value),
            PyAttributeInput::Tuple(name, value) => Attribute::new(name, value),
            PyAttributeInput::NameOnly(name) => Attribute::new(name, 1.0),
        }
    }
}

#[pyclass(module = "crfs", name = "Model")]
#[self_referencing]
struct PyModel {
    data: Vec<u8>,
    #[borrows(data)]
    #[covariant]
    model: Model<'this>,
}

#[pymethods]
impl PyModel {
    /// Create an instance of a model object from a model in memory
    #[new]
    fn new_py(data: Vec<u8>) -> PyResult<Self> {
        let model = PyModel::try_new(data, |data| Model::new(data))?;
        Ok(model)
    }

    /// Create an instance of a model object from a local file
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        let data = fs::read(path)?;
        Self::new_py(data)
    }

    /// Predict the label sequence for the item sequence.
    pub fn tag(&self, xseq: Vec<Vec<PyAttributeInput>>) -> PyResult<Vec<String>> {
        let mut tagger = self.borrow_model().tagger()?;
        let xseq: Vec<Vec<Attribute>> = xseq
            .into_iter()
            .map(|xs| xs.into_iter().map(Into::into).collect())
            .collect();
        let labels = tagger.tag(&xseq)?;
        Ok(labels.iter().map(|l| l.to_string()).collect())
    }

    /// Print the model in human-readable format
    pub fn dump(&self) -> PyResult<()> {
        let mut out = Vec::new();
        self.borrow_model().dump(&mut out)?;
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        handle.write_all(&out)?;
        Ok(())
    }
}

#[pymodule]
fn crfs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyAttribute>()?;
    m.add_class::<PyModel>()?;
    Ok(())
}
