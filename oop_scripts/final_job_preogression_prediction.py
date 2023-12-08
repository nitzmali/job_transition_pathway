#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[4]:


from tensorflow.keras.models import Model, load_model
import pickle
class loadjobModels:
    def __init__(self):
        pass

    def load_m(self):
        encoder_model = load_model('/Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/processed_datasets_v2/models/job/encoder_model.h5')

        # Save the decoder model
        decoder_model = load_model('/Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/processed_datasets_v2/models/job/decoder_model.h5')
        
        with open('/Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/processed_datasets_v2/models/job/tokenizer.pickle', 'rb') as handle:
            tokenizer_j_p = pickle.load(handle)
        return encoder_model,decoder_model,tokenizer_j_p

    def decode_sequence_job(self,input_seq):
        encoder_model,decoder_model,tokenizer_j_p=self.load_m()
        # Encode the input sequence to get the internal states
        states_value = encoder_model.predict(input_seq,verbose=0)
    
        # Generate empty target sequence of length 1 with only the start token
        target_seq = np.zeros((1, 1))
        start_token_index = tokenizer_j_p.word_index['start']  # Assuming you have a 'start' token
        target_seq[0, 0] = start_token_index
    
        # Sampling loop to generate sequence
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
    
            # Sample a token and add its corresponding word to the decoded sequence
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = tokenizer_j_p.index_word[sampled_token_index]
            if sampled_word != 'end':  # Assuming you have an 'end' token
                decoded_sentence += ' ' + sampled_word
    
            # Exit condition: either hit max length or find stop token
            if sampled_word == 'end' or len(decoded_sentence) > 50:
                stop_condition = True
    
            # Update the target sequence (of length 1) and states
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]
    
        return decoded_sentence



# In[7]:


tokenizer_j_e = AutoTokenizer.from_pretrained("bert-base-uncased")
model = TFAutoModel.from_pretrained("bert-base-uncased")
loadjobModels_cl = loadjobModels()


def get_embedding(text):
    inputs = tokenizer_j_e(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
    outputs = model(inputs)
    return tf.reduce_mean(outputs.last_hidden_state, 1).numpy()[0]  # Mean pooling

def batch_encode(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer_j_e(batch, return_tensors="tf", padding=True, truncation=True, max_length=512)
        outputs = model(inputs)
        batch_embeddings = tf.reduce_mean(outputs.last_hidden_state, 1).numpy()
        embeddings.extend(batch_embeddings)
    return embeddings



def predict_job(df_user):
    text_columns = ['Current Job', 'Next Job', 'Current Skills']
    embeddings_dict_user = {}
    for col in text_columns:
        embeddings_dict_user[col] = batch_encode(df_user[col].tolist())

    feature_columns = ['Current Job', 'Current Skills']
    # Stack embeddings horizontally (axis=1)
    user_embedding = np.hstack((embeddings_dict_user[each] for each in feature_columns)) 

    user_embedding.shape
    user_embedding = user_embedding.reshape(1, len(feature_columns), -1)

    X_user_tensor = tf.convert_to_tensor(user_embedding, dtype=tf.float32)
    next_job = loadjobModels_cl.decode_sequence_job(X_user_tensor[0:1])
    return  next_job.replace("start ","").lstrip().rstrip()
predict_job(df_user)


# In[9]:


from tensorflow.keras.models import Model, load_model
import pickle
class loadskillsModels:
    def __init__(self):
        pass

    def load_m(self):
        encoder_model_s = load_model('/Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/processed_datasets_v2/models/skills/encoder_model.h5')
        
        # Save the decoder model
        decoder_model_s = load_model('/Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/processed_datasets_v2/models/skills/decoder_model.h5')
        
        with open('/Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/processed_datasets_v2/models/skills/tokenizer.pickle', 'rb') as handle:
            tokenizer_s_p = pickle.load(handle)
        return encoder_model_s,decoder_model_s,tokenizer_s_p

    def decode_sequence_skills(self,input_seq):
        encoder_model_s,decoder_model_s,tokenizer_s_p = self.load_m()
        # Encode the input sequence to get the internal states
        states_value = encoder_model_s.predict(input_seq,verbose=0)
    
        # Generate empty target sequence of length 1 with only the start token
        target_seq = np.zeros((1, 1))
        start_token_index = tokenizer_s_p.word_index['start']  # Assuming you have a 'start' token
        target_seq[0, 0] = start_token_index
    
        # Sampling loop to generate sequence
        stop_condition = False
        decoded_sequence = []
        while not stop_condition:
            output_tokens, h, c = decoder_model_s.predict([target_seq] + states_value)
    
            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = tokenizer_s_p.index_word.get(sampled_token_index, '?')
    
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



# In[11]:


## load skills test data 

tokenizer_s_e = AutoTokenizer.from_pretrained("bert-base-uncased")
model = TFAutoModel.from_pretrained("bert-base-uncased")
loadskillsModels_cl = loadskillsModels()
def get_embedding(text):
    inputs = tokenizer_s_e(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
    outputs = model(inputs)
    return tf.reduce_mean(outputs.last_hidden_state, 1).numpy()[0]  # Mean pooling

def batch_encode(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer_s_e(batch, return_tensors="tf", padding=True, truncation=True, max_length=512)
        outputs = model(inputs)
        batch_embeddings = tf.reduce_mean(outputs.last_hidden_state, 1).numpy()
        embeddings.extend(batch_embeddings)
    return embeddings



#df = pd.read_csv("//Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/processed_datasets_v2/original_data/skills.csv")
#df_user = df[df["Current Job"]=="data scientist"][0:1]

def predict_skills(df_user):
    embeddings_dict_user_skills = {}
    text_columns = ['Current Job', 'Next Job', 'Current Skills']
    for col in text_columns:
        embeddings_dict_user_skills[col] = batch_encode(df_user[col].tolist())

    feature_columns = ['Current Job', 'Next Job','Current Skills']
    #print(embeddings_dict_user_skills.keys())
    # Stack embeddings horizontally (axis=1)
    user_embedding = np.hstack([embeddings_dict_user_skills[each] for each in feature_columns]) 

    user_embedding.shape
    user_embedding = user_embedding.reshape(1, len(feature_columns), -1)

    X_user_tensor = tf.convert_to_tensor(user_embedding, dtype=tf.float32)
    next_skills = loadskillsModels_cl.decode_sequence_skills(X_user_tensor[0:1]).strip().lstrip().rstrip()
    next_skills = next_skills.replace("start ","").replace("end ","").strip().lstrip().rstrip()
    return next_skills
predict_skills(df_user)


# In[23]:


import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class course_metrics:
    def __init__(self):
        pass
        
    def calculate_course_score(self,course, skill_gap, weight_skill_match=1, weight_subscribers=0.001, weight_rating=0.5):
        # Calculate the proportion of skill gap covered by course
        skills_matched = sum(skill in course['skills_tagged'] for skill in skill_gap)
        skill_match_score = skills_matched / len(skill_gap) if skill_gap else 0
    
        # Factor in course popularity and rating
        #subscriber_score = course['num_subscribers'] * weight_subscribers if 'num_subscribers' in course else 0
        #rating_score = course['rating'] * weight_rating if 'rating' in course else 0
    
        # Calculate total score
        total_score = (skill_match_score * weight_skill_match) #+ subscriber_score + rating_score
        return total_score
    

    
    def calculate_similarity_score(self,skill_gap,df_c_cleaned):
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(df_c_cleaned["full_description"])
    
            def preprocess_text(text):
                return " ".join(token.lemma_.lower() for token in nlp(text) if not token.is_punct and not token.is_stop)
        
            skill_gap_text = " ".join(skill_gap)
            # Vectorize the text
        
            skill_gap_vec = tfidf_vectorizer.transform([skill_gap_text])
            # Calculate cosine similarity
            cosine_similarities = cosine_similarity(skill_gap_vec, tfidf_matrix).flatten()
            df_c_cleaned["cosine_score"]=cosine_similarities
            return df_c_cleaned
    
    
    def select_relevant_course(self,skill_gap):
        # Define the course_prediction function
        df_c_cleaned = pd.read_csv("/Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/course_cleaned.csv")
        df_c_cleaned=df_c_cleaned.rename(columns={"Unnamed: 0":"course_number"})
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(df_c_cleaned["full_description"])
        # Apply the scoring function to each course
        #df_c_cleaned['course_score'] = df_c_cleaned.apply(calculate_course_score, axis=1, skill_gap=skill_gap)
        df_c_cleaned = self.calculate_similarity_score(skill_gap,df_c_cleaned)
        # Sort courses based on the calculated score
        #df_c_cleaned = df_c_cleaned.sort_values(by='course_score', ascending=False)
        df_c_cleaned = df_c_cleaned.sort_values(by='cosine_score', ascending=False)
        # Select the top course if available
        top_course = df_c_cleaned[["course_display_name","full_description"]].iloc[0] if not df_c_cleaned.empty else None
        top_10_course_number = df_c_cleaned["course_number"].iloc[0:10].tolist()
        top_10_course_number_score = {x:round(y,3) for x,y in zip(df_c_cleaned["course_number"].iloc[0:10].tolist(),df_c_cleaned["cosine_score"].iloc[0:10].tolist())}
        df_c_cleaned[["course_title",'skills_tagged','title','course_number']]
        top_10_title = set(df_c_cleaned["title"].iloc[0:10].tolist())
        top_10_course_title = set(df_c_cleaned["course_title"].iloc[0:10].tolist())
        all_skills_learnt = set(each for each in df_c_cleaned["course_title"].iloc[0:10].tolist())
        return top_10_title,all_skills_learnt,top_10_course_title

#


# In[24]:


# final job prediction


# In[28]:


def main():
    df_j = pd.read_csv("/Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/processed_datasets_v2/original_data/jobs.csv")
    df_user = df_j[df_j["Current Job"]=="data scientist"][0:1]
    
    df_c_cleaned = pd.read_csv("/Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/course_cleaned.csv")
    course_metrics_cl = course_metrics()
    top_course,top_10_course_number_score,top_10_course_number = course_metrics_cl.select_relevant_course(['python', 'power bi', 'sql'])
    
    
    
    current_job,final_job = "data analyst","big data engineer"
    current_skills,next_skills="",""
    predictions_dictionary = {}
    df_j = pd.read_csv("/Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/processed_datasets_v2/original_data/jobs.csv")
    i=0
    while current_job!=final_job and i<=5:
        df_user = df_j[df_j["Current Job"]==current_job]
        df_user = df_user.sample(n=1)
        #print(len(df_user))
        current_job = predict_job(df_user)
        i+=1
        current_skills = predict_skills(df_user)
        print("predicted Job: ",current_job)
        print("Predicted Skills: ",current_skills)
        predictions_dictionary[current_job]=current_skills
    
    
    pd.set_option('display.max_colwidth', None) 
    def course_prediction(skill_gap):
        df_c_cleaned = pd.read_csv("/Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/course_cleaned.csv")
        df_c_cleaned=df_c_cleaned.rename(columns={"Unnamed: 0":"course_number"})
        top_course, top_10_course_number_score, top_10_course_number = course_metrics_cl.select_relevant_course( skill_gap)
        return (top_course, top_10_course_number_score, top_10_course_number)
    
    predictions_df = pd.DataFrame(list(predictions_dictionary.items()), columns=['Predicted Job Title', 'Predicted Skills'])
    
    
    
    # Apply the function and create a DataFrame from the results
    course_predictions = predictions_df["Predicted Skills"].apply(lambda x: course_prediction(x.split(","))).tolist()
    courses_df = pd.DataFrame(course_predictions, columns=["top_10_predicted_course_category", "all_skills_learnt","top_10_predicted_course_title"])
    
    # Concatenate with the original DataFrame
    predictions_df = pd.concat([predictions_df, courses_df], axis=1)
    return predictions_df



# In[29]:


# Call the main function
if __name__ == "__main__":
    predictions_df = main()
    print(predictions_df)

# In[ ]:




