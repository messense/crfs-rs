/// Tuple of attribute and its value
///
/// This type is used for both training and prediction (tagging).
#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    /// Attribute name
    pub name: String,
    /// Value of the attribute
    pub value: f64,
}

impl Attribute {
    /// Create a new attribute with a name and value
    pub fn new<T: Into<String>>(name: T, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }
}

impl From<String> for Attribute {
    fn from(name: String) -> Self {
        Self { name, value: 1.0 }
    }
}

impl From<&str> for Attribute {
    fn from(name: &str) -> Self {
        Self {
            name: name.to_string(),
            value: 1.0,
        }
    }
}

impl<S: Into<String>> From<(S, f64)> for Attribute {
    fn from((name, value): (S, f64)) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }
}
