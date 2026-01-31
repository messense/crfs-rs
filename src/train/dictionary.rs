use std::collections::HashMap;

/// A bidirectional dictionary for mapping between strings and integer IDs
#[derive(Debug, Clone)]
pub struct Dictionary {
    /// Map from string to ID
    str_to_id: HashMap<String, u32>,
    /// Map from ID to string
    id_to_str: Vec<String>,
}

impl Dictionary {
    /// Create a new empty dictionary
    pub fn new() -> Self {
        Self {
            str_to_id: HashMap::new(),
            id_to_str: Vec::new(),
        }
    }

    /// Get the number of entries in the dictionary
    pub fn len(&self) -> usize {
        self.id_to_str.len()
    }

    /// Returns `true` if the dictionary contains no entries
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.id_to_str.is_empty()
    }

    /// Get or create an ID for a string
    /// Returns the ID for the string, creating a new entry if it doesn't exist
    pub fn get_or_insert(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.str_to_id.get(s) {
            id
        } else {
            let id = self.id_to_str.len() as u32;
            self.str_to_id.insert(s.to_string(), id);
            self.id_to_str.push(s.to_string());
            id
        }
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.str_to_id.clear();
        self.id_to_str.clear();
    }

    /// Iterate over all (string, id) pairs
    pub fn iter(&self) -> impl Iterator<Item = (&str, u32)> + '_ {
        self.id_to_str
            .iter()
            .enumerate()
            .map(|(id, s)| (s.as_str(), id as u32))
    }
}

impl Default for Dictionary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dictionary_basic() {
        let mut dict = Dictionary::new();
        assert_eq!(dict.len(), 0);

        let id1 = dict.get_or_insert("hello");
        assert_eq!(id1, 0);
        assert_eq!(dict.len(), 1);

        let id2 = dict.get_or_insert("world");
        assert_eq!(id2, 1);
        assert_eq!(dict.len(), 2);

        // Getting the same string should return the same ID
        let id3 = dict.get_or_insert("hello");
        assert_eq!(id3, id1);
        assert_eq!(dict.len(), 2);
    }

    #[test]
    fn test_dictionary_clear() {
        let mut dict = Dictionary::new();
        dict.get_or_insert("hello");
        dict.get_or_insert("world");
        assert_eq!(dict.len(), 2);

        dict.clear();
        assert_eq!(dict.len(), 0);
    }

    #[test]
    fn test_dictionary_iter() {
        let mut dict = Dictionary::new();
        dict.get_or_insert("hello");
        dict.get_or_insert("world");
        dict.get_or_insert("rust");

        let items: Vec<_> = dict.iter().collect();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], ("hello", 0));
        assert_eq!(items[1], ("world", 1));
        assert_eq!(items[2], ("rust", 2));
    }
}
