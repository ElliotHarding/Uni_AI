#######################################################
# Imports
#######################################################
import aiml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv
import sys
import inflect
import nltk
import wikipediaapi
import os
import re
import tensorflow as tf
import tensorflow_datasets as tfds
import _ddqn_for_guessing_game
import gym
import random

#######################################################
# Initialize wikipedia api
#######################################################
import wikipediaapi
wiki_wiki = wikipediaapi.Wikipedia('en')
wikipediaapi.log.setLevel(level=wikipediaapi.logging.ERROR)


#######################################################
# Reasoning vars
#######################################################
toyWorldString = """
field1 => f1
field2 => f2
field3 => f3
field4 => f4
leash => leash
bone => bone
the_lake => the_lake
barking => barking
walking => walking
sitting => sitting
lying_down => lying_down
rosie => rosie
bob => bob
dennis => dennis
spark => spark
charlie => charlie
max => max
rover => rover
dogs => {rosie, rover, bob, dennis, spark, charlie}
be_in => {}
trees => {}
grass => {}
is => {}
is_on => {}
is_chasing => {}
"""
locations = ["field1", "field2", "field3", "field4", "the_lake"]
actions = ["barking", "sitting", "walking", "lying_down"]
dogs = ["rosie", "bob", "dennis", "spark", "charlie", "max", "rover"]
canChase = dogs + ["bone"]
canBeOn = ["leash"]
canBePlaced = ["trees", "grass"] + canChase
folval = nltk.Valuation.fromstring(toyWorldString)
grammar_file = '_simple-sem.fcfg'
objectCounter = 0

#######################################################
# For singularization 
#######################################################
p = inflect.engine()

#######################################################
# Create a Kernel object & bootstrap to aiml file.
#######################################################
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="_chatbot-aiml.xml")

#######################################################
# Global vars
#######################################################
stopWords = ["the","is","an","a","at","and","i"]
waysOfSayingYes = ["yes", "y", "correct", "affirmative", "okay", "ok", "right", "of course", "by all means",
                   "sure", "indeed", "yea", "yeah", "yep", "yup", "certainly"]
breeds = []
breedInfo = []
sizes = []
NUM_BREEDS = None

transformer_model = None

guessing_game_model = _ddqn_for_guessing_game.DDQNAgent(np.array([1]), gym.spaces.Discrete(3))

MAX_SAMPLES = 50000 # Maximum number of samples to preprocess
MAX_LENGTH = 40 # Maximum sentence length
BATCH_SIZE = 64
BUFFER_SIZE = 20000

# Hyper-parameters
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1

EPOCHS = 0 #20

def preprocess_sentence(sentence):
  sentence = sentence.lower().strip()
  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)
  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
  sentence = sentence.strip()
  # adding a start and an end token to the sentence
  return sentence

def load_conversations():
  print("Loading conversations for transformer model...")
  path_to_zip = tf.keras.utils.get_file(
    'cornell_movie_dialogs.zip',
    origin=
    'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip',
    extract=True)

  path_to_dataset = os.path.join(os.path.dirname(path_to_zip), "cornell movie-dialogs corpus")

  path_to_movie_lines = os.path.join(path_to_dataset, 'movie_lines.txt')
  path_to_movie_conversations = os.path.join(path_to_dataset,
                                           'movie_conversations.txt')
  
  # dictionary of line id to text
  id2line = {}
  with open(path_to_movie_lines, errors='ignore') as file:
    lines = file.readlines()
  for line in lines:
    parts = line.replace('\n', '').split(' +++$+++ ')
    id2line[parts[0]] = parts[4]

  inputs, outputs = [], []
  with open(path_to_movie_conversations, 'r') as file:
    lines = file.readlines()
  for line in lines:
    parts = line.replace('\n', '').split(' +++$+++ ')
    # get conversation in a list of line ID
    conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
    for i in range(len(conversation) - 1):
      inputs.append(preprocess_sentence(id2line[conversation[i]]))
      outputs.append(preprocess_sentence(id2line[conversation[i + 1]]))
      if len(inputs) >= MAX_SAMPLES:
        return inputs, outputs
      
  return inputs, outputs

questions, answers = load_conversations()

print("Creating tokenizer for transformer model...")
# Build tokenizer using tfds for both questions and answers
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2**13)


# Define start and end token to indicate the start and end of a sentence
START_TOKEN, END_TOKEN = [tokenizer.vocab_size+1], [tokenizer.vocab_size + 2]
#print("START_TOKEN = ") 
#print(START_TOKEN)
#print("END_TOKEN = ") 
#print(END_TOKEN)
# Vocabulary size plus start and end token
VOCAB_SIZE = tokenizer.vocab_size + 3
#print("VOCAB_SIZE = ") 
#print(VOCAB_SIZE)


# Tokenize, filter and pad sentences
def tokenize_and_filter(inputs, outputs):
  tokenized_inputs, tokenized_outputs = [], []
  
  for (sentence1, sentence2) in zip(inputs, outputs):
    # tokenize sentence
    sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
    sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
    # check tokenized sentence max length
    if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
      tokenized_inputs.append(sentence1)
      tokenized_outputs.append(sentence2)
  
  # pad tokenized sentences
  tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
  tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_outputs, maxlen=MAX_LENGTH, padding='post')
  
  return tokenized_inputs, tokenized_outputs
  
questions, answers = tokenize_and_filter(questions, answers)

#print('Vocab size: {}'.format(VOCAB_SIZE))
#print('Number of samples: {}'.format(len(questions)))

# decoder inputs use the previous target as input
# remove START_TOKEN from targets
print("Creating dataset for transformer model...")
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1]
    },
    {
        'outputs': answers[:, 1:]
    },
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

print("Creating transformer model...")

def scaled_dot_product_attention(query, key, value, mask):
  """Calculate the attention weights. """
  matmul_qk = tf.matmul(query, key, transpose_b=True)

  # scale matmul_qk
  depth = tf.cast(tf.shape(key)[-1], tf.float32)
  logits = matmul_qk / tf.math.sqrt(depth)

  # add the mask to zero out padding tokens
  if mask is not None:
    logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k)
  attention_weights = tf.nn.softmax(logits, axis=-1)

  output = tf.matmul(attention_weights, value)

  return output
  
class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, name="multi_head_attention"):
    super(MultiHeadAttention, self).__init__(name=name)
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.query_dense = tf.keras.layers.Dense(units=d_model)
    self.key_dense = tf.keras.layers.Dense(units=d_model)
    self.value_dense = tf.keras.layers.Dense(units=d_model)

    self.dense = tf.keras.layers.Dense(units=d_model)

  def split_heads(self, inputs, batch_size):
    inputs = tf.reshape(
        inputs, shape=(batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])

  def call(self, inputs):
    query, key, value, mask = inputs['query'], inputs['key'], inputs[
        'value'], inputs['mask']
    batch_size = tf.shape(query)[0] 
    
    # linear layers
    query = self.query_dense(query)
    key = self.key_dense(key)
    value = self.value_dense(value)

    # split heads
    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    # scaled dot-product attention
    scaled_attention = scaled_dot_product_attention(query, key, value, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    # concatenation of heads
    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

    # final linear layer
    outputs = self.dense(concat_attention)

    return outputs

def create_padding_mask(x):
  mask = tf.cast(tf.math.equal(x, 0), tf.float32)
  # (batch_size, 1, 1, sequence length)
  return mask[:, tf.newaxis, tf.newaxis, :]
  
def create_look_ahead_mask(x):
  seq_len = tf.shape(x)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  padding_mask = create_padding_mask(x)
  return tf.maximum(look_ahead_mask, padding_mask)
 
class PositionalEncoding(tf.keras.layers.Layer):

  def __init__(self, position, d_model):
    super(PositionalEncoding, self).__init__()
    self.d_model = d_model
    self.pos_encoding = self.positional_encoding(position, d_model)

  def get_angles(self, position, i, d_model):
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(
        position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model=d_model)
    # apply sin to even index in the array
    sines = tf.math.sin(angle_rads[:, 0::2])
    # apply cos to odd index in the array
    cosines = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, tf.float32)

  def call(self, inputs):
    return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
    
def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  attention = MultiHeadAttention(
      d_model, num_heads, name="attention")({
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': padding_mask
      })
  attention = tf.keras.layers.Dropout(rate=dropout)(attention)
  attention = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs + attention)

  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention + outputs)

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)
          
def encoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  for i in range(num_layers):
    outputs = encoder_layer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name="encoder_layer_{}".format(i),
    )([outputs, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)
         
def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name="look_ahead_mask")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

  attention1 = MultiHeadAttention(
      d_model, num_heads, name="attention_1")(inputs={
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': look_ahead_mask
      })
  attention1 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention1 + inputs)

  attention2 = MultiHeadAttention(
      d_model, num_heads, name="attention_2")(inputs={
          'query': attention1,
          'key': enc_outputs,
          'value': enc_outputs,
          'mask': padding_mask
      })
  attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
  attention2 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention2 + attention1)
      
  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(outputs + attention2)

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)
    
def decoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
  inputs = tf.keras.Input(shape=(None,), name='inputs')
  enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name='look_ahead_mask')
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
  
  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  for i in range(num_layers):
    outputs = decoder_layer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name='decoder_layer_{}'.format(i),
    )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)
    
def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

  enc_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(inputs)
  # mask the future tokens for decoder inputs at the 1st attention block
  look_ahead_mask = tf.keras.layers.Lambda(
      create_look_ahead_mask,
      output_shape=(1, None, None),
      name='look_ahead_mask')(dec_inputs)
  # mask the encoder outputs for the 2nd attention block
  dec_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='dec_padding_mask')(inputs)

  enc_outputs = encoder(
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[inputs, enc_padding_mask])
  
  dec_outputs = decoder(
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

  outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

  return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name) 
    
def loss_function(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  
  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)

  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def accuracy(y_true, y_pred):
  # ensure labels have shape (batch_size, MAX_LENGTH - 1)
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

def get_model():
  tf.keras.backend.clear_session()

  model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)
  
  model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
  return model  

#Load transformer model
transformer_model = get_model()

print("Loading transformer model weights...")
transformer_model.load_weights('C:/Users/elliot/Documents/Github/transformer_weights/transformer_model')
transformer_model.fit(dataset, epochs=EPOCHS)


#######################################################
# ResetToyWorld
#######################################################
def ResetToyWorld():
    global folval
    folval = nltk.Valuation.fromstring(toyWorldString)
    
#######################################################
# ReadFiles
#   
#   Reads breed-and-information file into global arrays
#   breeds & breedInfo
#######################################################
def ReadFiles():
    global breeds, breedInfo, sizes
    try:        
        breedAndInfoPairs = list(csv.reader(open('breed-and-information.csv', 'r')))
                       
        breeds = [row[0] for row in breedAndInfoPairs]
        breedInfo = [row[1] for row in breedAndInfoPairs]
        sizes = [row[2] for row in breedAndInfoPairs]
        
        print("Loading guessing game model weights...")
        guessing_game_model.load_weights('C:/Users/elliot/Documents/Github/ddqn_weights/ddqn_gameGuesser.h5')
        
        NUM_BREEDS = len(sizes)
        
    except (IOError) as e:
        print("Error occured opening one of the files! " + e.strerror)
        sys.exit()

#######################################################
# GetSimilarityArray
#   
#######################################################
def GetSimilarityArray(string, searchArray):
    array = [string] + searchArray
    tfidf = TfidfVectorizer(stop_words=stopWords).fit_transform(array)
    similarityArray = cosine_similarity(tfidf[0:1], tfidf)
    similarityArray = np.delete(similarityArray, 0)

    return similarityArray

#######################################################
# GetIndexOfMostSimilar
#   Checks input list for most similar string to input string
#   Returns -1 if no strings over similarityBound
#######################################################
def GetIndexOfMostSimilar(string, searchArray, similariyBound=0):
    similarityArray = GetSimilarityArray(string, searchArray)
    if similarityArray[similarityArray.argmax()] < similariyBound:
        return -1
    else:
        return similarityArray.argmax()

#######################################################
# Exit
#######################################################
def Exit():
    print("Bye!")
    sys.exit()

#######################################################
# GetInput
#   Handles & returns user input
#######################################################
def GetInput():
    try:
        userInput = input("> ")
        if userInput == '':
            return "-"
    except (KeyboardInterrupt, EOFError) as e:
        Exit()
    return userInput

#######################################################
# CheckSimilarDogs
#
#
#######################################################
def CheckSimilarDogs(userInput, arrayToCheck, similariyBound):

    dogsAndSimilarity = GetSimilarityArray(userInput, arrayToCheck)

    dogsToCheck = ""
    index = 0
    for i in dogsAndSimilarity:
        if i > similariyBound:
            dogsToCheck += breeds[index] + ", "
        index+=1

    arrayLen = len(dogsToCheck)
    dogsToCheck = dogsToCheck[0:arrayLen-2]

    if arrayLen > 0:
        print("You can ask me about any of these dogs: " + dogsToCheck)
        return 1
    return 0

#######################################################
# WikiSearch
#######################################################
def WikiSearch(search):    
    wpage = wiki_wiki.page(search)
    if wpage.exists():
        print(wpage.summary)
        print("Learn more at", wpage.canonicalurl)
    else:
        print("Sorry, I don't know what that is.")
            
#######################################################
# DescribeDog
# describes a dog
#######################################################
def DescribeDog(dogName):
    iMostSimilarDog = GetIndexOfMostSimilar(dogName, breeds, 0.8)        
    if iMostSimilarDog != -1:
        print(breedInfo[iMostSimilarDog])
        
    else:
        HandleUnknownInput(dogName)
            
#######################################################
# HandleUnknownInput
#
#   Attempts to find figure out if user is asking for
#   information on a dog
#######################################################
def HandleUnknownInput(search):

    if CheckSimilarDogs(search, breeds, 0.3) == 1:
        return

    #If it's a single word, just quick check if its plural
    if not " " in search:
        singular = p.singular_noun(search)
        if singular:            
            if CheckSimilarDogs(p.singular_noun(search), breeds, 0.3) == 1:
                return

    if CheckSimilarDogs(search, breedInfo, 0.3) == 1:
        return

    #No point in doing a wiki search if related to bot
    if " you" in search or "you " in search or " me" in search or "me " in search:
        return 0
        
    print(evaluate(search))
    return 0

#######################################################
# PrintDogSize
#
#   Prints size of a dog
#######################################################
def PrintDogSize(dogName):
    iMostSimilarDog = GetIndexOfMostSimilar(dogName, breeds, 0.8)
    if iMostSimilarDog != -1:
        if sizes[iMostSimilarDog] == "L":
            print("A " + breeds[iMostSimilarDog] + " is a large sized dog.")
            return
        elif sizes[iMostSimilarDog] == "M":
            print("A " + breeds[iMostSimilarDog] + " is a medium sized dog")
            return
        elif sizes[iMostSimilarDog] == "S":
            print("A " + breeds[iMostSimilarDog] + " is a small sized dog")
            return
    print("Sorry I don't know the size of that dog.")

#######################################################
# ListSizedDogs
#
#   Prints list of dogs of specified size
#######################################################
def ListSizedDogs(size):
    breedList = ""
    index = 0
    for dog in breeds:
        if sizes[index] == size:
            breedList += dog + ", "
        index+=1

    #Get rid of last ", "
    breedList = breedList[0:len(breedList)-2]
    print("Here's the dogs I found: " + breedList)

#######################################################
# IsCrossBreed
#
#   Returns if a dog at index iDog in breedInfo is a cross breed or not
#######################################################
def IsCrossBreed(iDog):
    dogDescription = breedInfo[iDog]
    if "mixed breed" in dogDescription or "a cross" in dogDescription:
        return 1
    else:
        return 0

#######################################################
# PrintCrossBreed
#
#   Prints out if a dog is a cross breed or not
#######################################################
def PrintCrossBreed(dog):
    iMostSimilarDog = GetIndexOfMostSimilar(dog, breeds, 0.5)
    if iMostSimilarDog != -1:        
        if IsCrossBreed(iMostSimilarDog):
            print("A " + breeds[iMostSimilarDog] + " is in fact a cross breed.")
        else:
            print("A " + breeds[iMostSimilarDog] + " is in fact a pure breed.")
        return
            
    if CheckSimilarDogs(dog, breeds, 0.3) == 1:
        return
    else:
        print("It seems I couldn't find what you were looking for. Would you like me to wikipekida search '" + dog + "'?")
    
 
#######################################################
# PrintCrossBreeds
#
#   Prints out all cross breeds
#######################################################
def PrintCrossBreeds():
    breedList = ""
    index = 0
    for dog in breeds:
        if IsCrossBreed(index):
            breedList += dog + ", "
        index+=1

    #Get rid of last ", "
    breedList = breedList[0:len(breedList)-2]
    print("Here are all the cross breeds: " + breedList)

def ClearEmptyFolvalSlot(slot):
    if len(folval[slot]) == 1: 
        if ('',) in folval[slot]:
            folval[slot].clear()

def GetMatchingFolvalValues(linkWord, searchWord):
    return nltk.Model(folval.domain, folval).satisfiers(nltk.Expression.fromstring(linkWord+"(x," + searchWord + ")"), "x", nltk.Assignment(folval.domain))

def CheckCorrectInput(filter, toCheck):
    for item in filter:
        if item == toCheck:
            return True
    return False

#Set values/actions/positions
def SetFolValValues(data):
    global objectCounter
    if data[0] in folval:
        if data[2] in folval:    

            if data[1] == "be_in" and (CheckCorrectInput(locations, data[0]) or not CheckCorrectInput(locations, data[2])):
                print("Sorry, either " + data[2] + " is not a location, or " + data[0] + " is one.")
                return
                         
            if data[1] == "is" and (not CheckCorrectInput(actions, data[2]) or not CheckCorrectInput(dogs, data[0])):
                print("Sorry, either " + data[2] + " is not an action, or " + data[0] + " is not a dog.")
                return

            if data[1] == "is_chasing" and (not CheckCorrectInput(dogs, data[0]) or not CheckCorrectInput(canChase, data[2])):
                print("Sorry, either " + data[0] + " is not a dog, or a " + data[2] + " can't be chased.")
                return

            if data[1] == "is_on" and (not CheckCorrectInput(canBeOn, data[2]) or not CheckCorrectInput(dogs, data[0])):
                print("Sorry, either " + data[2] + " can't be applied to a dog, or " + data[0] + " is not a dog.")
                return

            o = data[0]
                
            #Can't be in same place or doing two actions at once
            for item in folval[data[1]]:
                if data[0] in item:
                    if data[1] == "be_in":
                        print(data[0] + " has moved to " + data[2])
                    elif data[1] == "is":                    
                        print(data[0] + " is now " + data[2])                     
                    folval[data[1]].remove(item)
                    break 

            if data[0] == "trees" or data[0] == "grass":              
                o = 'o' + str(objectCounter)
                objectCounter += 1
                folval['o' + o] = o
                ClearEmptyFolvalSlot(data[0])
                folval[data[0]].add((o,))                  

            ClearEmptyFolvalSlot(data[1])
            folval[data[1]].add((o, folval[data[2]]))   
        
            print("Done.")
        else:
            print(data[2] + " does not exist in toy world.")
    else:
        print(data[0] + " does not exist in toy world.")

#Yes/no queries
def FolValYesNoQueries(data):
    try:
        sent = " "
        results = nltk.evaluate_sents([sent.join(data).lower()], grammar_file, nltk.Model(folval.domain, folval), nltk.Assignment(folval.domain))[0][0]
        if results[2] == True:
            print("Yes.")
        else:
            print("No.")
    except:
        print("Sorry, I don't know that.")

#Query ~ List all values which meet x
def FolValListMatches(data):
    try:
        sat = GetMatchingFolvalValues(data[0], data[1])
        if len(sat) == 0:
            print(data[2])
        else:   
            for so in sat:
                if any(char.isdigit() for char in so) == True:                
                    for k, v in folval.items():
                        if len(v) > 0:
                            vl = list(v)
                            if len(vl[0]) == 1:
                                for i in vl:
                                    if i[0] == so:
                                        print(k)
                                        break
                else:
                    print(so)
    except:
        print("Sorry can't help with that")

#Query ~ do any values meet x
def FolValAnyMeet(data):
    if data[1] == "lying down":
        data[1] = "lying_down"

    if data[1] in folval:
        if len(GetMatchingFolvalValues(data[0], data[1])) == 0:
            print("No.")
        else:    
            print("yes.")
    else:
        print("That action is not in the toy world.")   

def DescribeToyWorld():
    print("The toy world contains: ")
    print("Locations: " + ', '.join(locations))    
    print("Dogs: " + ', '.join(dogs))
    print("Actions for dogs: " + ', '.join(actions))
    print("Chaseable objects: " + ', '.join(canChase))
    print("Placeable objects: " + ', '.join(canBePlaced))
    print("Try the command: Put rover in field1")
    

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
  
  sentence = preprocess_sentence(sentence)
  sentence = tf.expand_dims(START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)

  for i in range(MAX_LENGTH):
    predictions = transformer_model(inputs=[sentence, output], training=False)

    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    output = tf.concat([output, predicted_id], axis=-1)

  prediction = tf.squeeze(output, axis=0)
  predicted_sentence = tokenizer.decode(
    [i for i in prediction if i < tokenizer.vocab_size])
  
  return predicted_sentence
  
#######################################################
# Game_IsCorrectChoice ~ Returns weather or not answer 
# to guessing game is correct
#######################################################
def Game_IsCorrectChoice(choice, answer):
    choice = choice.lower()

    if choice == "small" or choice == "s":
        return (answer == "S")
    elif choice == "medium" or choice == "m":
        return (answer == "M")
    elif choice == "large" or choice == "L":
        return (answer == "L")
    else:
        return False
        
#######################################################
# Game_BotGuess ~ Returns bots guess to answer of guessing game
#######################################################
def Game_BotGuess(dogIndex):
    guess = guessing_game_model.act(np.reshape(np.array([dogIndex]), [1, 1]))
    if guess == 0:
        print("Guess : Small")
        return "Small"
    elif guess == 1:
        print("Guess : Medium")
        return "Medium"
    else:
        print("Guess : Large")
        return "Large"

#######################################################
# Game_MatchGameMainLoop 
#######################################################
def Game_MatchGameMainLoop(breeds, sizes, numOfBreeds, userIsPlaying):

    print("Aim of the game is to guess the size of a dog. (Small, Medium, Large). Type 'quit' to stop.")

    gamesCounter = 0
    while True:

        dogIndex = random.randrange(0, numOfBreeds)
        print()
        print("Guess the size of a " + breeds[dogIndex])
        
        playerInput = None
        if userIsPlaying:
            playerInput = GetInput()
        else:
            playerInput = Game_BotGuess(dogIndex)
            gamesCounter += 1
        
        if playerInput == "quit":
            print("See you next time!")
            break        
        elif Game_IsCorrectChoice(playerInput, sizes[dogIndex]):
            print("Correct!")
        else:
            print("Incorrect!")
            
        if gamesCounter == 10:
            break


#######################################################
# HandleAIMLCommand
#
#   Handles responses for AIML commands
#######################################################
def HandleAIMLCommand(cmd, data):
    
    if cmd == 0:
        Exit()                
    elif cmd == 1:
        DescribeDog(data[0])
    elif cmd == 2:
        WikiSearch(data[0])
    elif cmd == 3:
        PrintCrossBreed(data[0])
    elif cmd == 4:
        PrintCrossBreeds()
    elif cmd == 5:
        PrintDogSize(data[0])
    elif cmd == 6:
        ListSizedDogs(data[0])
    elif cmd == 7: 
        SetFolValValues(data)
    elif cmd == 8:
        FolValYesNoQueries(data)
    elif cmd == 9:
        FolValListMatches(data)  
    elif cmd == 10:
        DescribeToyWorld()  
    elif cmd == 11:
        FolValAnyMeet(data)
    elif cmd == 12:
        ResetToyWorld()
        print("Done.")
    elif cmd == 13:
        Game_MatchGameMainLoop(breeds, sizes, len(breeds), True)
    elif cmd == 14:
        Game_MatchGameMainLoop(breeds, sizes, len(breeds), False)
    elif cmd == 99:
        HandleUnknownInput(data[0])

#######################################################
# Main loop
#######################################################
def MainLoop():    
    print(("\nHi! I'm the dog breed information chatbot.\n - Try asking me a question about a specifc breed. \n - Ask me about groups of breeds(hounds, terriers, retrievers).\n - Try and describe a breed for me to guess. \n - Ask me to tell you a dog related joke.\n - Or ask me about the toy world.\n"))
    while True: 
    
        #Get input
        userInput = GetInput()
        
        #Get response from aiml
        answer = kern.respond(userInput)

        #Check if command response
        if userInput != "-" and answer != "" and answer[0] == '#':
            
            #Split answer into cmd & input
            params = answer[1:].split('$')
            HandleAIMLCommand(int(params[0]), params[1:])

        #Otherwise direct respond
        else:
            print(answer)
            
ReadFiles()
MainLoop()