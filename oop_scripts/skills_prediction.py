#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import os 
import numpy as np 
import tensorflow as tf
# Set a random seed for reproducibility
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
import pandas as pd 
from transformers import AutoTokenizer, TFAutoModel
import pandas as pd 
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


# In[3]:


class loaddata:
    def __init__(self):
        pass 

    ## Load data 
    def load_d(self):
        synthetic_user_data = pd.read_csv("/Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/synthetic_data_cleaned_v5.csv")
        import ast 
        #synthetic_user_data=synthetic_user_data[:100]
        df = synthetic_user_data.copy()
        # Fill missing values if necessary
        df.fillna({'Course Taken': 'no course','course title': 'no course'}, inplace=True)
        df["Course Taken"]=df["Course Taken"].apply(lambda x:x.replace("NAN","No Course"))
        
        #df["Course Skills"]=df["Course Skills"].apply(lambda x:ast.literal_eval(x))
        df["Course Skills"].apply(lambda x: x if len(x)!=0 else "no skills")
        #df["Course title"]=df["Course title"].apply(lambda x:x.replace("NAN","No Course"))
        def set_to_string(skill_set):
            if not skill_set:  # Checks if the set is empty
                return 'no skills' 
            # Check if skill_set is a string that needs to be evaluated
            if isinstance(skill_set, str):
                try:
                    # Try to evaluate the string as a set
                    skill_set = ast.literal_eval(skill_set)
                except (ValueError, SyntaxError):
                    # Handle cases where the string is not a valid set
                    pass  # You might want to return a default value or handle this case as needed
            # Convert to string if it's a set or list
            if isinstance(skill_set, (set, list)):
                return ', '.join(skill_set)
            return skill_set 
        
        set_columns = ['Current Skills', 'Next Skills', 'Skill Gap', 'Course Skills']
        for col in set_columns:
            print(col)
            #df[col]=df[col].apply(lambda x:ast.literal_eval(x))
            df[col] = df[col].apply(set_to_string)
        
        
        
        
        
        #df = pd.read_csv("/Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/df_embedding_bert.csv")
        
        # Loading the JSON file back into a dictionary
        import json
        Job_progression_dictionary_file_path = "/Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/job_progression_dictionary.json"
        
        with open(Job_progression_dictionary_file_path, 'r') as json_file:
            sorted_full_job_progression_dict_lower = json.load(json_file)
        
        #print(sorted_full_job_progression_dict_lower)
        # Output: {'name': 'John', 'age': 30, 'city': 'New York'}

        df = df.drop(columns=[ 'Unnamed: 0'])
        df = df[df["random next job"]!=1].drop_duplicates()
        df = df[['Current Job', 'Next Job', 'Current Skills']].drop_duplicates()
        return df,sorted_full_job_progression_dict_lower
        





# In[16]:


ld_cl = loaddata()
df,sorted_full_job_progression_dict_lower = ld_cl.load_d()


# In[4]:


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = TFAutoModel.from_pretrained("bert-base-uncased")
class create_embedding:
    def __init__(self):
        pass

    def batch_encode(texts, batch_size=32):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="tf", padding=True, truncation=True, max_length=512)
            outputs = model(inputs)
            batch_embeddings = tf.reduce_mean(outputs.last_hidden_state, 1).numpy()
            embeddings.extend(batch_embeddings)
        return embeddings
    #text_columns = ['Current Job', 'Next Job', 'Current Skills', 'Next Skills', 'Skill Gap', 'course title', 'Course Skills', 'Course Taken']
    text_columns = ['Current Job', 'Next Job', 'Current Skills']
    embeddings_dict = {}
    for col in text_columns:
        embeddings_dict[col] = batch_encode(df[col].tolist())



# In[ ]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[ ]:


#job prediction

# load embeddings 

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = TFAutoModel.from_pretrained("bert-base-uncased")
class prep_data:
    def __init__(self):
        pass
    def prepare_data(self,df,tokenizer,model):
        # Add 'start' and 'end' tokens to each target sequence
    # Add 'start' and 'end' tokens to each target sequence
    df['Next Skills'] = df['Next Skills'].apply(lambda x: 'start,' + x + ',end')
    
    
    loaded_embeddings = np.load('/Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/processed_datasets_v2/embeddings_data/all_embeddings_skills_only.npz')
        
        #'Next Skills', 'Skill Gap', 'course title', 'Course Skills'
        feature_columns = ['Current Job', 'Current Skills', 'Next Job' ]
        # Stack embeddings horizontally (axis=1)
        all_embeddings = np.hstack((loaded_embeddings[each] for each in feature_columns)) # Add other embeddings as needed
        
        tokenizer = Tokenizer(split=',')
        tokenizer.fit_on_texts(df['Next Skills'])  # Replace with your target column
        target_seqs = tokenizer.texts_to_sequences(df['Next Skills'])
        target_seqs_padded = pad_sequences(target_seqs, padding='post')
        
        # Vocabulary size for the output
        target_vocab_size = len(tokenizer.word_index) + 1
        
        # Shift target sequences for the decoder's training
        decoder_input_data = target_seqs_padded[:, :-1]  # all except last token
        decoder_target_data = target_seqs_padded[:, 1:]  # all except first token
        
        
        #y_course = loaded_embeddings['Course Taken_embedding']
        
        from sklearn.model_selection import train_test_split
        
        # Assuming 'target' is your target array
        X_train, X_test, y_job_train, y_job_test = train_test_split(all_embeddings, target_seqs_padded, test_size=0.2, random_state=42)
        decoder_input_train, decoder_input_test = train_test_split(decoder_input_data, test_size=0.2, random_state=42)
        decoder_target_train, decoder_target_test = train_test_split(decoder_target_data, test_size=0.2, random_state=42)
        
        # Reshape input embeddings for the model
        # This depends on the expected input shape of your model. 
        # For example, if your model expects 3D input (samples, timesteps, features):
        X_train_reshaped = X_train.reshape(X_train.shape[0], len(feature_columns), -1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], len(feature_columns), -1)
        
        import tensorflow as tf
        
        X_train_tensor = tf.convert_to_tensor(X_train_reshaped, dtype=tf.float32)
        X_test_tensor = tf.convert_to_tensor(X_test_reshaped, dtype=tf.float32)
        decoder_input_train_tensor = tf.convert_to_tensor(decoder_input_train, dtype=tf.float32)
        decoder_input_test_tensor = tf.convert_to_tensor(decoder_input_test, dtype=tf.float32)
        decoder_target_train_tensor = tf.convert_to_tensor(decoder_target_train, dtype=tf.float32)
        decoder_target_test_tensor = tf.convert_to_tensor(decoder_target_test, dtype=tf.float32)
        
        
        # Verify the new shapes
        print("X_train_tensor shape:", X_train_tensor.shape)
        print("X_test_tensor shape:", X_test_tensor.shape)
        
        print("decoder_input_train_tensor shape:", decoder_input_train_tensor.shape)
        print("decoder_input_test_tensor shape:", decoder_input_test_tensor.shape)
        print("decoder_target_train_tensor shape:", decoder_target_train_tensor.shape)
        print("decoder_target_test_tensor shape:", decoder_target_test_tensor.shape)

        return X_train_tensor,X_test_tensor,decoder_input_train_tensor,decoder_input_test_tensor,decoder_target_train_tensor,\
                decoder_target_test_tensor


# In[ ]:





# In[ ]:





# In[5]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding

class define_model:
    def __init__(self):
        pass

    def model_creation(self,feature_columns):
        # Model parameters
        bert_embedding_dim = 768
        latent_dim = 256  # Dimensionality of the LSTM layer
        target_vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size for the output
        
        # Encoder
        encoder_inputs = Input(shape=(len(feature_columns), bert_embedding_dim))  # 2 features, each with a BERT embedding of size 768
        encoder_lstm = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = Input(shape=(None,))  # 'None' allows the model to handle variable length sequences
        decoder_embedding = Embedding(target_vocab_size, latent_dim)
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_dense = Dense(target_vocab_size, activation='softmax')
        
        # Embed and decode the sequence
        dec_emb = decoder_embedding(decoder_inputs)
        decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # Define the model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        # Compile the model
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

        return model

def decode_sequence(input_seq):
    # Encode the input sequence to get the internal states
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1 with only the start token
    target_seq = np.zeros((1, 1))
    start_token_index = tokenizer.word_index['start']  # Assuming you have a 'start' token
    target_seq[0, 0] = start_token_index

    # Sampling loop to generate sequence
    stop_condition = False
    decoded_sequence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, '?')

        # Check for 'end' token or maximum length
        if sampled_word == 'end' or len(decoded_sequence) > 50:  # 'end' is your end-of-sequence token
            stop_condition = True
        else:
            decoded_sequence.append(sampled_word)

        # Update the target sequence (of length 1) and states
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    # Join the words with a comma
    decoded_sentence = ', '.join(decoded_sequence)
    return decoded_sentence




# In[6]:


def main():
    prep_data_cl = prep_data()
    X_train_tensor,X_test_tensor,decoder_input_train_tensor,decoder_input_test_tensor,decoder_target_train_tensor,\
                    decoder_target_test_tensor = prep_data_cl.prepare_data(df,tokenizer,model)
    md_cl =define_model()
    model = md_cl.model_creation(feature_columns)
    
    # Train the model
    history = model.fit(
        [X_train_tensor, decoder_input_train_tensor], 
        np.expand_dims(decoder_target_train_tensor, -1),  # Add an extra dimension to the target
        batch_size=32,
        epochs=20,
        validation_data=(
            [X_test_tensor, decoder_input_test_tensor], 
            np.expand_dims(decoder_target_test_tensor, -1)
        )
    )
    
    
    model.summary()
    
    # Encoder model for inference
    encoder_model = Model(encoder_inputs, encoder_states)
    
    # Decoder setup for inference
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    dec_emb2 = decoder_embedding(decoder_inputs)  # Reuse the same embedding layer
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_outputs2)
    
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs2] + decoder_states2
    )
    
    for i in range(100):  # Generate predictions for the first 10 test samples
        input_seq = X_test_tensor[i: i + 1]
        decoded_sentence = decode_sequence(input_seq)
        print('Predicted sequence:', decoded_sentence)
    
    
    from sklearn.metrics import accuracy_score
    
    def strip_target_variable(target_value):
        target_value=target_value.replace("start","").replace("end","").lstrip().rstrip()
        return target_value
    # Generate predictions for the test set
    y_pred = [decode_sequence(X_test_tensor[i: i + 1]) for i in range(len(X_test_tensor))]
    
    # You need to have your test target sequences in text format for comparison
    y_true = [" ".join(tokenizer.sequences_to_texts([y])) for y in y_job_test]
    
    y_pred = list(map(strip_target_variable,y_pred))
    y_true = list(map(strip_target_variable,y_true))
    # Compute accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)
    
    ## save the models for prediction 
    
    from tensorflow.keras.models import Model, load_model
    
    # Assuming encoder_model and decoder_model are already defined as shown in your snippet
    
        
    ## save the models for prediction 
    
    from tensorflow.keras.models import Model, load_model
    
    # Assuming encoder_model and decoder_model are already defined as shown in your snippet
    
    
    # Save the encoder model
    encoder_model.save('/Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/processed_datasets_v2/models/skills/encoder_model.h5')
    
    # Save the decoder model
    decoder_model.save('/Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/processed_datasets_v2/models/skills/decoder_model.h5')
    
    import pickle
    
    # Assuming 'tokenizer' is your tokenizer object
    with open('/Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/processed_datasets_v2/models/skills/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)







# In[ ]:


# Call the main function
if __name__ == "__main__":
    main()

