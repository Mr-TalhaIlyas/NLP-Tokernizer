# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 11:45:40 2022

@author: talha
"""
#%%
import collections
import os
import pathlib
import re
import string
import sys
import tempfile
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
pwd = pathlib.Path.cwd()
#%%
# Load your data files
path2pt = 'C:/Users/talha/Desktop/tokenizer/ds/pt_train.txt'
path2en = 'C:/Users/talha/Desktop/tokenizer/ds/en_train.txt'

pt_train = pathlib.Path(path2pt).read_text(encoding="utf-8").splitlines()
en_train = pathlib.Path(path2en).read_text(encoding="utf-8").splitlines()

# convert the list to tensor
pt_train = tf.convert_to_tensor(pt_train, dtype=tf.string)
en_train = tf.convert_to_tensor(en_train, dtype=tf.string)

# make dataset
pt_data = tf.data.Dataset.from_tensor_slices(pt_train)
en_data = tf.data.Dataset.from_tensor_slices(en_train)
#%%
# importing and initilize BERT tokenizer params

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

bert_tokenizer_params=dict(lower_case=True)
reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]

bert_vocab_args = dict(
    # The target vocabulary size
    vocab_size = 8000,
    # Reserved tokens that must be included in the vocabulary
    reserved_tokens=reserved_tokens,
    # Arguments for `text.BertTokenizer`
    bert_tokenizer_params=bert_tokenizer_params,
    # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
    learn_params={},
    )

#%%
'''
Generate vocab
'''
pt_vocab = bert_vocab.bert_vocab_from_dataset(
    pt_data.batch(1000).prefetch(2),
    **bert_vocab_args
)

en_vocab = bert_vocab.bert_vocab_from_dataset(
    en_data.batch(1000).prefetch(2),
    **bert_vocab_args
)
# print some values in vocab
print(en_vocab[100:110])
#%%
'''
Wrtie vocab files
'''
def write_vocab_file(filepath, vocab):
  with open(filepath, 'w', encoding="utf-8") as f:
    for token in vocab:
      print(token, file=f)
      
write_vocab_file('pt_vocab.txt', pt_vocab)
write_vocab_file('en_vocab.txt', en_vocab)
#%%
'''
give bert tokernizer the vocab of your dataset and initilize the tokenizer methods.
'''
pt_tokenizer = text.BertTokenizer('pt_vocab.txt', **bert_tokenizer_params)
en_tokenizer = text.BertTokenizer('en_vocab.txt', **bert_tokenizer_params) # bert_tokenizer_params=dict(lower_case=True)

#%%
'''
tokenize your text
'''
# get an exmple string from data and pass it to tokenizer
en_examples = []
for example in en_data.as_numpy_iterator():
    en_examples.append(example)
    break

# Tokenize the examples -> (batch, word, word-piece)
token_batch = en_tokenizer.tokenize(en_examples) # 'hello there !' test it!!!
print(f'Token Batch shape : {token_batch.shape} => [batch, word, word-piece]')
# Merge the word and word-piece axes -> (batch, tokens)
token_batch = token_batch.merge_dims(-2,-1)
print(f'Token Batch reshaped : {token_batch.shape} => [batch, tokens]')
print(f'Tokenized list of input string : {token_batch.to_list}')
#%%
'''
detokenize your text
'''
words = en_tokenizer.detokenize(token_batch)
tf.strings.reduce_join(words, separator=' ', axis=-1)
print(f'Detokenized words: {words}')

#%%

'''
But for transformer training we have to add the START and END sentense tokens so...
'''
START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
END = tf.argmax(tf.constant(reserved_tokens) == "[END]")

def add_start_end(ragged):
  count = ragged.bounding_shape()[0]
  starts = tf.fill([count,1], START)
  ends = tf.fill([count,1], END)
  return tf.concat([starts, ragged, ends], axis=1)

words = en_tokenizer.detokenize(add_start_end(token_batch))
tf.strings.reduce_join(words, separator=' ', axis=-1)
print(words)
'''
Also when we are printintg text we don't want START END tokens to be shown on the output so...
'''
def cleanup_text(reserved_tokens, token_txt):
  # Drop the reserved tokens, except for "[UNK]".
  bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
  bad_token_re = "|".join(bad_tokens)

  bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
  result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

  # Join them into strings.
  result = tf.strings.reduce_join(result, separator=' ', axis=-1)

  return result

token_batch = en_tokenizer.tokenize(en_examples).merge_dims(-2,-1)
print(token_batch)
words = en_tokenizer.detokenize(add_start_end(token_batch))
print(words)
clean = cleanup_text(reserved_tokens, words).numpy()
print(clean)

#%%
class CustomTokenizer(tf.Module):
  def __init__(self, reserved_tokens, vocab_path):
    self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
    self._reserved_tokens = reserved_tokens
    self._vocab_path = tf.saved_model.Asset(vocab_path) # save the assets in the model

    vocab = pathlib.Path(vocab_path).read_text(encoding="utf-8").splitlines()
    self.vocab = tf.Variable(vocab)

    ## Create the signatures for export:   

    # Include a tokenize signature for a batch of strings. 
    self.tokenize.get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string))

    # Include `detokenize` and `lookup` signatures for:
    #   * `Tensors` with shapes [tokens] and [batch, tokens]
    #   * `RaggedTensors` with shape [batch, tokens]
    self.detokenize.get_concrete_function(
        tf.TensorSpec(shape=[None, None], dtype=tf.int64))
    self.detokenize.get_concrete_function(
          tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

    self.lookup.get_concrete_function(
        tf.TensorSpec(shape=[None, None], dtype=tf.int64))
    self.lookup.get_concrete_function(
          tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

    # These `get_*` methods take no arguments
    self.get_vocab_size.get_concrete_function()
    self.get_vocab_path.get_concrete_function()
    self.get_reserved_tokens.get_concrete_function()

  @tf.function
  def tokenize(self, strings):
    enc = self.tokenizer.tokenize(strings)
    # Merge the `word` and `word-piece` axes.
    enc = enc.merge_dims(-2,-1)
    enc = add_start_end(enc)
    return enc

  @tf.function
  def detokenize(self, tokenized):
    words = self.tokenizer.detokenize(tokenized)
    return cleanup_text(self._reserved_tokens, words)

  @tf.function
  def lookup(self, token_ids):
    return tf.gather(self.vocab, token_ids)

  @tf.function
  def get_vocab_size(self):
    return tf.shape(self.vocab)[0]

  @tf.function
  def get_vocab_path(self):
    return self._vocab_path

  @tf.function
  def get_reserved_tokens(self):
    return tf.constant(self._reserved_tokens)

#%%
'''
Initilize tf.Module and save the model by given params
'''
tokenizers = tf.Module()
tokenizers.pt = CustomTokenizer(reserved_tokens, 'pt_vocab.txt')
tokenizers.en = CustomTokenizer(reserved_tokens, 'en_vocab.txt')

model_name = 'my_pt_en_convertor'
tf.saved_model.save(tokenizers, model_name)
#%%

'''
Relode for sanity check
'''
reloaded_tokenizers = tf.saved_model.load(model_name)
print(reloaded_tokenizers.en.get_vocab_size().numpy())

tokens = reloaded_tokenizers.en.tokenize(['Hello TensorFlow!'])
print(tokens.numpy())

text_tokens = reloaded_tokenizers.en.lookup(tokens)
print(text_tokens)

round_trip = reloaded_tokenizers.en.detokenize(tokens)

print(round_trip.numpy()[0].decode('utf-8'))

#%%