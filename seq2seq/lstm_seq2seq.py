import pandas as pd
import numpy as np
import re

# ## Load saved sequences data

# In[80]:

df_seq = pd.read_csv('../data/cat113_root_name.csv')
X = df_seq.iloc[:,0].values
y = df_seq.iloc[:,1].values
X_roots = df_seq.iloc[:,2].values


# ## Processing data

# In[81]:

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


# In[82]:

X = pre_process(X)
X_roots = pre_process(X_roots)


# In[83]:

# preview data
for title, root in zip(X[:5], X_roots[:5]):
    print(title, '-', root)


# ## Sequence model

# ### Embedding

# In[84]:

num_samples = len(X)


# In[108]:

input_texts = []
target_texts = []

input_tokens = set()
target_tokens = set()


# In[109]:

for i in range(num_samples):
    # cast into tokens
    input_texts.append(X[i])
    target_texts.append('\t' + X_roots[i] + '\n')
    
    for word in input_texts[i]:
        if word not in input_tokens:
            input_tokens.add(word)
    for word in target_texts[i]:
        if word not in target_tokens:
            target_tokens.add(word)

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


# In[117]:

# building dictionary of tokens
input_token_index = dict([(token, i) for i, token in enumerate(input_tokens)])
target_token_index = dict([(token, i) for i, token in enumerate(target_tokens)])


# In[118]:

# building embedding for input and target data
encoder_input_data = np.zeros((num_samples, max_encoder_seq_length, num_encoder_tokens))
decoder_input_data = np.zeros((num_samples, max_decoder_seq_length, num_decoder_tokens))
decoder_target_data = np.zeros((num_samples, max_decoder_seq_length, num_decoder_tokens))


# In[119]:

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for j, word in enumerate(input_text):
        encoder_input_data[i, j, input_token_index[word]] = 1.
    for j, word in enumerate(target_text):
        decoder_input_data[i, j, target_token_index[word]] = 1.
        
        # decoder_target_data is ahead of decoder_input_data by one timestep
        if j > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, j - 1, target_token_index[word]] = 1.
            


# ### Model

# In[106]:

from keras.models import Model
from keras.layers import Input, LSTM, Dense


# In[124]:

latent_dim = 256
epochs = 1
batch_size = 128


# In[121]:

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]


# In[122]:

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.summary()

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('s2s.h5')

