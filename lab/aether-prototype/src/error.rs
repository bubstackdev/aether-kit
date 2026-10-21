/// Errors keep native failures available to callers through `source()`.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error("unsupported model configuration: {0}")]
    Unsupported(String),
    #[error("generation cancelled between model calls")]
    Cancelled,
    #[error(transparent)]
    Runtime(#[from] corsairs::Error),
    #[error("{stage}: {source}")]
    Stage {
        stage: String,
        #[source]
        source: corsairs::Error,
    },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("PNG encoding: {0}")]
    Png(#[from] png::EncodingError),
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn require(condition: bool, message: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::Invalid(message.into()))
    }
}
