/// An attribute consists of an attribute id with its value
#[derive(Debug, Clone, Copy)]
pub struct Attribute {
    /// Attribute id
    pub id: u32,
    /// Value of the attribute
    pub value: f64,
}

/// An item consists of an array of attributes
pub type Item = Vec<Attribute>;

/// An instance consists of a sequence of items and labels
#[derive(Debug, Clone)]
pub struct Instance {
    /// Number of items/labels in the sequence
    pub num_items: u32,
    /// Array of the item sequence
    pub items: Vec<Item>,
    /// Array of the label sequence
    pub labels: Vec<u32>,
    /// Instance weight
    pub weight: f64,
    /// Group ID of the instance
    pub group: u32,
}

impl Attribute {
    pub fn new(id: u32, value: f64) -> Self {
        Self { id, value }
    }
}

impl Instance {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            num_items: 0,
            items: Vec::with_capacity(cap),
            labels: Vec::with_capacity(cap),
            weight: 1.0,
            group: 0,
        }
    }

    pub fn push(&mut self, item: Item, label: u32) {
        self.items.push(item);
        self.labels.push(label);
        self.num_items += 1;
    }
}
