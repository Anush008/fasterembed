import test from 'ava'

import { sum, TextEmbedding, EmbeddingModel } from '../index.js'

test('sum from native', (t) => {
  t.is(sum(1, 2), 3)
})

test('embedding fastembed-rs', (t) => {

  const model = new TextEmbedding(EmbeddingModel.AllMiniLML6V2);

  let documents = [
    "passage: Hello, World!",
    "query: Hello, World!",
    "passage: This is an example passage.",
    // You can leave out the prefix but it's recommended
    "fastembed-js is licensed under MIT"
  ];

  const embeddings = model.embed(documents, 2);

  t.is(embeddings.length, documents.length);
})