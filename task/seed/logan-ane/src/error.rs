use std::fmt;

#[derive(Debug)]
pub enum AneError {
    UnsupportedPlatform,
    FrameworkLoad {
        path: &'static str,
        message: String,
    },
    MissingClass(&'static str),
    MissingSelector {
        class: &'static str,
        selector: &'static str,
    },
    AbiMismatch {
        class: &'static str,
        selector: &'static str,
        expected: &'static str,
        actual: String,
    },
    ObjectiveC {
        operation: &'static str,
        message: String,
    },
    NullResult(&'static str),
    Surface {
        operation: &'static str,
        code: i32,
    },
    InvalidArgument(String),
    Io(std::io::Error),
}

impl fmt::Display for AneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => write!(
                f,
                "Apple Neural Engine access requires macOS on Apple Silicon"
            ),
            Self::FrameworkLoad { path, message } => {
                write!(f, "failed to load private framework {path}: {message}")
            }
            Self::MissingClass(name) => {
                write!(f, "private Objective-C class {name} is unavailable")
            }
            Self::MissingSelector { class, selector } => {
                write!(f, "private API selector {class}::{selector} is unavailable")
            }
            Self::AbiMismatch {
                class,
                selector,
                expected,
                actual,
            } => write!(
                f,
                "private API ABI mismatch for {class}::{selector}: expected {expected}, got {actual}"
            ),
            Self::ObjectiveC { operation, message } => {
                write!(f, "ANE {operation} failed: {message}")
            }
            Self::NullResult(operation) => write!(f, "ANE {operation} returned nil"),
            Self::Surface { operation, code } => {
                write!(f, "IOSurface {operation} failed with IOReturn {code}")
            }
            Self::InvalidArgument(message) => write!(f, "invalid ANE argument: {message}"),
            Self::Io(error) => write!(f, "ANE filesystem I/O failed: {error}"),
        }
    }
}

impl std::error::Error for AneError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for AneError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

pub type Result<T> = std::result::Result<T, AneError>;
