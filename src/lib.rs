#![deny(clippy::all)]

#[macro_use]
extern crate napi_derive;

use std::path::{Path, PathBuf};

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

const DEFAULT_MAX_LENGTH: i32 = 512;
const DEFAULT_CACHE_DIR: &str = ".fastembed_cache";
const DEFAULT_EMBEDDING_MODEL: JSEmbeddingModel = JSEmbeddingModel::BGESmallENV15;
const DEFAULT_SHOW_DOWNLOAD_PROGRESS: bool = true;

#[napi]
pub fn sum(a: i32, b: i32) -> i32 {
  a + b
}

#[napi(js_name = "TextEmbedding")]
pub struct JsTextEmbedding {
  embedding: TextEmbedding,
}

#[napi(js_name = "EmbeddingModel")]
pub enum JSEmbeddingModel {
  AllMiniLML6V2,
  BGEBaseENV15,
  BGELargeENV15,
  BGESmallENV15,
  NomicEmbedTextV1,
  ParaphraseMLMiniLML12V2,
}

impl From<JSEmbeddingModel> for EmbeddingModel {
  fn from(model: JSEmbeddingModel) -> Self {
    match model {
      JSEmbeddingModel::AllMiniLML6V2 => EmbeddingModel::AllMiniLML6V2,
      JSEmbeddingModel::BGEBaseENV15 => EmbeddingModel::BGEBaseENV15,
      JSEmbeddingModel::BGELargeENV15 => EmbeddingModel::BGELargeENV15,
      JSEmbeddingModel::BGESmallENV15 => EmbeddingModel::BGESmallENV15,
      JSEmbeddingModel::NomicEmbedTextV1 => EmbeddingModel::NomicEmbedTextV1,
      JSEmbeddingModel::ParaphraseMLMiniLML12V2 => EmbeddingModel::ParaphraseMLMiniLML12V2,
    }
  }
}

#[napi]
impl JsTextEmbedding {
  #[napi(constructor)]
  pub fn new(
    model: Option<JSEmbeddingModel>,
    cache_dir: Option<String>,
    max_length: Option<i32>,
    show_download_progress: Option<bool>,
  ) -> Self {
    let model_name: EmbeddingModel = model.unwrap_or(DEFAULT_EMBEDDING_MODEL).into();
    let cache_dir: String = cache_dir.unwrap_or(DEFAULT_CACHE_DIR.to_string());
    let cache_dir: PathBuf = Path::new(&cache_dir).to_path_buf();
    let max_length: usize = max_length.unwrap_or(DEFAULT_MAX_LENGTH) as usize;
    let show_download_progress = show_download_progress.unwrap_or(DEFAULT_SHOW_DOWNLOAD_PROGRESS);

    let options: InitOptions = InitOptions {
      model_name,
      cache_dir,
      max_length,
      show_download_progress,
      ..Default::default()
    };
    JsTextEmbedding {
      embedding: TextEmbedding::try_new(options).unwrap(),
    }
  }

  #[napi]
  pub fn embed(&self, texts: Vec<&str>, batch_size: Option<i32>) -> Vec<Vec<f32>> {
    self
      .embedding
      .embed(texts, batch_size.map(|x| x as usize))
      .unwrap()
  }
}