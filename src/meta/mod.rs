
mod vector_utils;
pub use self::vector_utils::*;



use std::error::Error;
use std::fmt::{self, Formatter, Display, Debug};
use std::convert::From;
use std::path::Path;
use std::io::{self, Read, Write};
use std::fs::{File, OpenOptions};
use std::f32;
use std::num;


pub type CubrainResult<T> = Result<T, CubrainError>;

pub struct CubrainError {
    description: String,
}
impl Display for CubrainError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}
impl Debug for CubrainError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}
impl Error for CubrainError {
    fn description(&self) -> &str {
        &self.description
    }
    fn cause(&self) -> Option<&Error> {
        None
    }
}
impl From<io::Error> for CubrainError {
    fn from(obj: io::Error) -> CubrainError {
        CubrainError { description: obj.description().to_string() }
    }
}
impl From<num::ParseFloatError> for CubrainError {
    fn from(obj: num::ParseFloatError) -> CubrainError {
        CubrainError { description: obj.description().to_string() }
    }
}
impl CubrainError {
    pub fn new(description: String) -> CubrainError {
        CubrainError { description }
    }
}


pub trait Codec {
    type OutputType;
    fn encode(&self) -> String;
    fn decode(data: &str) -> CubrainResult<Self::OutputType>;
}
pub trait Save: Codec {
    fn save(&self, file_path: &str) -> CubrainResult<()> {
        let path = Path::new(file_path);
        let mut file = OpenOptions::new().create(true).write(true).open(path)?;
        file.set_len(0)?;
        file.write(self.encode().as_bytes())?;
        Ok(())
    }
    fn load(file_path: &str) -> CubrainResult<<Self as Codec>::OutputType> {
        let path = Path::new(file_path);
        let mut file : File = File::open(path)?;
        let mut data = String::new();
        file.read_to_string(&mut data)?;
        Self::decode(data.as_str())
    }
}


impl Codec for [f32] {
    type OutputType = Vec<f32>;

    fn encode(&self) -> String {
        let mut out = String::with_capacity(self.len());
        for i in 0..self.len()-1 {
            out.push_str(&format!("{:.25} ", self[i]));
        }
        out.push_str(&format!("{:.25}", self[self.len()-1]));
        out
    }
    fn decode(data: &str) -> CubrainResult<Vec<f32>> {
        let mut output = Vec::new();
        for value in data.split_whitespace() {
            let value = value.parse::<f32>()?;
            output.push(value);
        }
        Ok(output)
    }
}
impl Save for [f32] {}

impl<T: Codec> Save for T {
    fn save(&self, file_path: &str) -> CubrainResult<()> {
        let path = Path::new(file_path);
        let mut file = OpenOptions::new().create(true).write(true).open(path)?;
        file.set_len(0)?;
        file.write(self.encode().as_bytes())?;
        Ok(())
    }
    fn load(file_path: &str) -> CubrainResult<<Self as Codec>::OutputType> {
        let path = Path::new(file_path);
        let mut file : File = File::open(path)?;
        let mut data = String::new();
        file.read_to_string(&mut data)?;
        Self::decode(data.as_str())
    }
}