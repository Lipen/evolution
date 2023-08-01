use std::collections::HashMap;
use std::fs::{File, OpenOptions};

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};

use crate::summary::Summary;

#[derive(Serialize, Deserialize)]
pub struct Cache {
    pub data: HashMap<String, Summary>,
}

impl Cache {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    // Load the cache from a file
    pub fn load_from_file(file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(file_path)?;
        let mut decoder = GzDecoder::new(file);
        let config = bincode::config::standard();
        let cache = bincode::serde::decode_from_std_read(&mut decoder, config)?;
        Ok(cache)
    }

    // Save the cache into a file
    pub fn save_to_file(&self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(file_path)?;
        let mut encoder = GzEncoder::new(file, Compression::default());
        let config = bincode::config::standard();
        bincode::serde::encode_into_std_write(self, &mut encoder, config)?;
        Ok(())
    }
}

impl Cache {
    pub fn get(&self, key: &String) -> Option<&Summary> {
        self.data.get(key)
    }

    pub fn insert(&mut self, key: String, value: Summary) {
        self.data.insert(key, value);
    }
}
