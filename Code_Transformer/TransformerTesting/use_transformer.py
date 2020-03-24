from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.random.set_seed(1234)

#!pip install tensorflow-datasets==1.2.0
import tensorflow_datasets as tfds

import os
import re

model = tf.keras.models.load_model('transformer_model.h5')

#######################################################
# Preprocess sentence ~ format sentence for processing in transformer model
#######################################################
def preprocess_sentence(sentence):
  sentence = sentence.lower().strip()
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)
  sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
  sentence = sentence.strip()
  return sentence

#######################################################
# Predict ~ Return result of input sentence using transformer model
#######################################################
def evaluate(sentence):
  START_TOKEN = 8331
  END_TOKEN = 8332
  VOCAB_SIZE = 8333
  
  sentence = preprocess_sentence(sentence)
  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)

  for i in range(MAX_LENGTH):
    predictions = model(inputs=[sentence, output], training=False)

    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    output = tf.concat([output, predicted_id], axis=-1)

  prediction = tf.squeeze(output, axis=0)
  predicted_sentence = tokenizer.decode(
    [i for i in prediction if i < tokenizer.vocab_size])
  
  return predicted_sentence
