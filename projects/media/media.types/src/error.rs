//! Core error type with stable machine-readable codes.

use std::fmt;

/// Every failure surfaced by the core carries a code so hosts can branch on it.
#[derive(Debug)]
pub struct Error {
    /// Stable code, e.g. `invalid_scene`.
    pub code: &'static str,
    /// Human-readable detail.
    pub message: String,
}

impl Error {
    /// Scene description failed validation or parsing.
    #[must_use]
    pub fn invalid(message: impl Into<String>) -> Self {
        Self { code: "invalid_scene", message: message.into() }
    }

    /// An input contract (sample rate, channels, dimensions) was violated.
    #[must_use]
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self { code: "invalid_input", message: message.into() }
    }

    /// An I/O step (fonts, output file) failed.
    #[must_use]
    pub fn io(message: impl Into<String>) -> Self {
        Self { code: "io", message: message.into() }
    }

    /// The encoder backend failed.
    #[must_use]
    pub fn encode(message: impl Into<String>) -> Self {
        Self { code: "encode", message: message.into() }
    }

    /// Imported media could not be opened or decoded.
    #[must_use]
    pub fn media(message: impl Into<String>) -> Self {
        Self { code: "media", message: message.into() }
    }

    /// A capability is not implemented on this platform.
    #[must_use]
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self { code: "unsupported_platform", message: message.into() }
    }

    /// The stable code for hosts.
    #[must_use]
    pub fn code(&self) -> &'static str {
        self.code
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::io(value.to_string())
    }
}
