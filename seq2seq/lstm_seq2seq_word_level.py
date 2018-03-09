
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import re
from time import time


# ## Load saved sequences data

# In[2]:

df_seq = pd.read_csv('../data/cat661-root-train.csv')
X = df_seq.iloc[:,0].values
y = df_seq.iloc[:,1].values
X_roots = df_seq.iloc[:,2].values


# ## Processing data

# In[3]:

# pre-processing
def pre_process(X):
    X_p = []

    for name in X:
        name = name.lower().split()
        name = [re.compile('[(),]+').sub('', w) for w in name] 
        name = [w for w in name if re.compile('[\W_]+').sub('', w)] # remove all words that only constain special character
        name = ' '.join(name)
        #name = ViTokenizer.tokenize(name)
        X_p.append(name)

    return X_p


# In[4]:

X = pre_process(X)
X_roots = pre_process(X_roots)


# In[5]:

# preview data
for title, root in zip(X[:5], X_roots[:5]):
    print(title, '-', root)


# ## Sequence model

# ### Embedding

# In[6]:

num_samples = len(X)


# In[14]:

input_texts = []
target_texts = []

input_tokens = set()
target_tokens = set()

w2d = {}


# In[15]:

# word-level tokens
for i in range(num_samples):
    # cast into tokens
    input_texts.append(X[i].split())
    target_texts.append(['\START_'] + X_roots[i].split() + ['\END_'])
    
    for word in input_texts[i]:
        if word not in input_tokens:
            input_tokens.add(word)
        if word not in w2d:
            w2d[word] = 1
        else:
            w2d[word] += 1
            
    for word in target_texts[i]:
        if word not in target_tokens:
            target_tokens.add(word)
        if word not in w2d:
            w2d[word] = 1
        else:
            w2d[word] += 1

input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))

num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

max_encoder_seq_length = max([len(seq) for seq in input_texts])
max_decoder_seq_length = max([len(seq) for seq in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)  


# In[34]:

w2d_sorted = sorted(w2d.items(), key=lambda item: item[1])


# In[45]:

threshold = 20

print('Root tokens length:', len(w2d_sorted))

# apply thres hold
w2d_thres = [item for item in w2d_sorted if item[1] > threshold]

print('Threshold tokens length:', len(w2d_thres))
w2d_thres[:100]


# In[46]:

# building dictionary of tokens
input_token_index = dict([(token, i) for i, token in enumerate(input_tokens)])
target_token_index = dict([(token, i) for i, token in enumerate(target_tokens)])


# In[49]:

# building embedding for input and target data
encoder_input_data = np.zeros((num_samples, max_encoder_seq_length))
decoder_input_data = np.zeros((num_samples, max_decoder_seq_length))
decoder_target_data = np.zeros((num_samples, max_decoder_seq_length, num_decoder_tokens))


# In[119]:

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for j, word in enumerate(input_text):
        encoder_input_data[i, j] = input_token_index[word]
    for j, word in enumerate(target_text):
        decoder_input_data[i, j] = target_token_index[word]
        
        # decoder_target_data is ahead of decoder_input_data by one timestep
        if j > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, j - 1, target_token_index[word]] = 1.
            


# ### Model

# In[50]:

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.callbacks import TensorBoard


# In[51]:

latent_dim = 256
batch_size = 128


# In[53]:

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,))
encoder_eb = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_eb)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]


# In[54]:

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_eb = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_eb, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


# In[55]:

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.summary()


# In[126]:
import sys
if __name__ == '__main__':
    epochs = 1
    param = sys.argv[1:]
    if len(param) > 0:
        epochs = int(param[0])
    
    # Run training
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              callbacks=[TensorBoard(log_dir="logs/{}".format(time()))])
    # Save model
    model.save('s2s_word.h5')
"""
# In[56]:

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\START_']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)


# In[ ]:
"""

