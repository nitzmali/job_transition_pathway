#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import json

class DataLoader:
    def __init__(self):
        self.df_c_cleaned = None
        self.df_j_cleaned = None
        self.df_s_skills_list = None
        self.sorted_full_job_progression_dict_lower = None

    def load_course_data(self, filepath):
        self.df_c_cleaned = pd.read_csv(filepath)
        self.df_c_cleaned = self.df_c_cleaned.rename(columns={"Unnamed: 0": "course_number"})

    def load_job_data(self, filepath):
        self.df_j_cleaned = pd.read_csv(filepath)
        self.df_j_cleaned = self.df_j_cleaned[self.df_j_cleaned["title_cleaned"] != "unmatched"]

    def load_skills_data(self, filepath):
        self.df_s_skills_list = pd.read_csv(filepath)

    def load_job_progression_dict(self, filepath):
        with open(filepath, 'r') as json_file:
            self.sorted_full_job_progression_dict_lower = json.load(json_file)

    # Other data loading methods as needed...
    
    
    
class DataManipulator:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def get_skills_list(self):
        return self.data_loader.df_s_skills_list["skills_list"].unique().tolist()

    def select_random_job(self, jobs_df):
        random_idx = random.randint(0, len(jobs_df) - 1)
        random_job = jobs_df.iloc[random_idx]
        return random_job.title_cleaned, random_job.skills_tagged, random_job.normalized_min, random_job.normalized_max

    def select_next_job(self, current_job, jobs_df, set_job_unique, job_progression_dict):
        # Implementation of select_next_job method...
        next_job,next_skills,random_job_selection_flag="","",0
        # If the current job is in the progression dict, select the next job from there
        if current_job in job_progression_dict:
            next_jobs = job_progression_dict[current_job]
            #next_jobs=list(set_job_unique.intersection(set(next_jobs)))
            if len(next_jobs)!=0:
                next_job_flag = True
                i=0
                while next_job_flag and i<len(next_jobs):
                    if next_jobs[i] in set_job_unique:
                        next_job = next_jobs[i]
                        next_job_flag=False
                    i+=1
                if next_job_flag==False:
                #next_job = random.choice(next_jobs)
                #print(next_job, " choosen")
                    try:
                        #next_skills = jobs_df[jobs_df['title_cleaned'] == next_job]['skills_tagged'].values[0]
                        next_skills_rows = jobs_df[jobs_df['title_cleaned'] == next_job]
                        if not next_skills_rows.empty:
                            random_row = next_skills_rows.sample(n=1)
                            next_skills = random_row['skills_tagged'].values[0]
                    except:
                        next_job = random.choice(next_jobs)
                        #next_skills = jobs_df[jobs_df['title_cleaned'] == next_job]['skills_tagged'].values[0]
                        next_skills_rows = jobs_df[jobs_df['title_cleaned'] == next_job]
                        if not next_skills_rows.empty:
                            random_row = next_skills_rows.sample(n=1)
                            next_skills = random_row['skills_tagged'].values[0]
                else:
                    # If not, fall back to random selection for this example
                    next_job, next_skills,_,_ = select_random_job(jobs_df)
                    random_job_selection_flag=1
            else:
                next_job, next_skills,_,_ = select_random_job(jobs_df)
                random_job_selection_flag=1

        else:
            next_job, next_skills,_,_ = select_random_job(jobs_df)
            random_job_selection_flag=1
        return next_job, next_skills,random_job_selection_flag
        pass

    def set_to_string(self, skill_set):
        # Implementation of set_to_string method...
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
        pass



# In[3]:


class DataGenerator:
    def __init__(self, data_loader, data_manipulator):
        self.data_loader = data_loader
        self.data_manipulator = data_manipulator

    def generate_synthetic_data(self, num_samples=10):
        # Implementation of synthetic data generation...
        synthetic_data = []
        set_job_unique = set(jobs_df["title_cleaned"].unique())
        for _ in range(num_samples):
            #select_random_job(jobs_df)
            current_job, current_skills,current_min_salary,current_max_salary = select_random_job(jobs_df)
            # Select a job to progress to that requires at least one additional skill
            next_job, next_skills,random_job_selection_flag = select_next_job(current_job,jobs_df,set_job_unique,sorted_full_job_progression_dict_lower)
            #while set(next_skills).issubset(current_skills):
            #    next_job, next_skills = select_random_job(jobs_df)
            next_skills=ast.literal_eval(next_skills)
            current_skills=ast.literal_eval(current_skills)
            # Find the skill gap
            skill_gap = list(set(next_skills) - set(current_skills))
            #print(skill_gap)
            if  len(skill_gap)==0:
                skill_gap = list(next_skills)
            # Find a course that teaches one of the skills in the gap
            course,top_10_courses_score,top_10_courses_number = select_relevant_course(courses_df, skill_gap)
            #print(course)
            if course is not None:
                course_name = course.course_title
                course_skills = course.skills_tagged
                course_title = course.title
            else:
                # If no single course covers the gap, select a random one for this synthetic example
                course_name = "Random Course"
                course_skills = "Random Skill"

            if len(current_skills) and len(next_skills):
                # Append to the synthetic data list
                synthetic_data.append({
                    'Current Job': current_job,
                    'Next Job': next_job,
                    'Current Skills': current_skills,
                    'Next Skills': next_skills,
                    'Skill Gap': skill_gap,
                    'Current Min Salary':current_min_salary,
                    'Current Max Salary':current_max_salary,
                    'course title':course_title,
                    'Course Taken': course_name,
                    'Course Skills': course_skills,
                    'top_10_course_numbers':top_10_courses_number,
                    'top_10_course_score':top_10_courses_score,
                    'random next job':random_job_selection_flag
                })

    
        return pd.DataFrame(synthetic_data)
        pass

    def generate_synthetic_next_job_data(self, synthetic_user_data):
        # Implementation of synthetic next job data generation...
        synthetic_data = []
        set_job_unique = set(jobs_df["title_cleaned"].unique())
        for i,iterrow in synthetic_user_data.iterrows():
            #print(i, iterrow["Next Job"],iterrow["Next Skills"])
            #select_random_job(jobs_df)
            current_job, current_skills,current_min_salary,current_max_salary = iterrow["Next Job"],(iterrow["Next Skills"]),iterrow["Current Min Salary"],iterrow["Current Max Salary"]
            # Select a job to progress to that requires at least one additional skill
            next_job, next_skills,random_job_selection_flag = select_next_job(current_job,jobs_df,set_job_unique,sorted_full_job_progression_dict_lower)
            #while set(next_skills).issubset(current_skills):
            #    next_job, next_skills = select_random_job(jobs_df)
            next_skills=ast.literal_eval(next_skills)
            # Find the skill gap
            skill_gap = list(set(next_skills) - set(current_skills))
            #print(skill_gap)
            # Find a course that teaches one of the skills in the gap
            course,top_10_courses_score,top_10_courses_number = select_relevant_course(courses_df, skill_gap)
            #print(course)
            if course is not None:
                course_name = course.course_title
                course_skills = course.skills_tagged
                course_title = course.title
            else:
                # If no single course covers the gap, select a random one for this synthetic example
                course_name = "Random Course"
                course_skills = "Random Skill"

            if len(current_skills) and len(next_skills):
                # Append to the synthetic data list
                synthetic_data.append({
                    'Current Job': current_job,
                    'Next Job': next_job,
                    'Current Skills': current_skills,
                    'Next Skills': next_skills,
                    'Skill Gap': skill_gap,
                    'Current Min Salary':current_min_salary,
                    'Current Max Salary':current_max_salary,
                    'course title':course_title,
                    'Course Taken': course_name,
                    'Course Skills': course_skills,
                    'top_10_course_numbers':top_10_courses_number,
                    'top_10_course_score':top_10_courses_score,
                    'random next job':random_job_selection_flag
                })
    
        return pd.DataFrame(synthetic_data)
        pass

    # Other generation methods...
    


# In[4]:


import spacy
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import pandas as pd

# Utility functions for text processing, course selection, etc.
def preprocess_text(text):
    doc = nlp(text)
    return " ".join(token.lower_ for token in doc if not token.is_punct and not token.is_stop)

def calculate_course_score(course, skill_gap, weight_skill_match=1, weight_subscribers=0.001, weight_rating=0.5):
    # Implementation of calculate_course_score function...
    skills_matched = sum(skill in course['skills_tagged'] for skill in skill_gap)
    skill_match_score = skills_matched / len(skill_gap) if skill_gap else 0

    # Factor in course popularity and rating
    #subscriber_score = course['num_subscribers'] * weight_subscribers if 'num_subscribers' in course else 0
    #rating_score = course['rating'] * weight_rating if 'rating' in course else 0

    # Calculate total score
    total_score = (skill_match_score * weight_skill_match) #+ subscriber_score + rating_score
    return total_score
    pass

def calculate_similarity_score(skill_gap, df_c_cleaned):
    # Implementation of calculate_similarity_score function...
    def preprocess_text(text):
        return " ".join(token.lemma_.lower() for token in nlp(text) if not token.is_punct and not token.is_stop)

    skill_gap_text = " ".join(skill_gap)
    # Vectorize the text

    skill_gap_vec = tfidf_vectorizer.transform([skill_gap_text])
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(skill_gap_vec, tfidf_matrix).flatten()
    df_c_cleaned["cosine_score"]=cosine_similarities
    return df_c_cleaned
    pass

def select_relevant_course(df_c_cleaned, skill_gap):
    # Implementation of select_relevant_course function...
    # Apply the scoring function to each course
    df_c_cleaned['course_score'] = df_c_cleaned.apply(calculate_course_score, axis=1, skill_gap=skill_gap)
    df_c_cleaned = calculate_similarity_score(skill_gap,df_c_cleaned)
    # Sort courses based on the calculated score
    #df_c_cleaned = df_c_cleaned.sort_values(by='course_score', ascending=False)
    df_c_cleaned = df_c_cleaned.sort_values(by='cosine_score', ascending=False)
    # Select the top course if available
    top_course = df_c_cleaned.iloc[0] if not df_c_cleaned.empty else None
    top_10_course_number = df_c_cleaned["course_number"].iloc[0:10].tolist()
    top_10_course_number_score = {x:round(y,3) for x,y in zip(df_c_cleaned["course_number"].iloc[0:10].tolist(),df_c_cleaned["cosine_score"].iloc[0:10].tolist())}
    return top_course,top_10_course_number_score,top_10_course_number
    pass


# In[5]:


# Main execution
data_loader = DataLoader()
data_loader.load_course_data("/Users/uzairpachhapure/Library/CloudStorage/GoogleDrive-uzairriyaz.pachhapure@sjsu.edu/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/course_cleaned.csv")
data_loader.load_job_data("/Users/uzairpachhapure/Library/CloudStorage/GoogleDrive-uzairriyaz.pachhapure@sjsu.edu/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/job_cleaned_salary.csv")
data_loader.load_skills_data("/Users/uzairpachhapure/Library/CloudStorage/GoogleDrive-uzairriyaz.pachhapure@sjsu.edu/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/skills_dataset/skills_list_df.csv")
data_loader.load_job_progression_dict("/Users/uzairpachhapure/Library/CloudStorage/GoogleDrive-uzairriyaz.pachhapure@sjsu.edu/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/job_progression_dictionary.json")

data_manipulator = DataManipulator(data_loader)
data_generator = DataGenerator(data_loader, data_manipulator)

# Generate synthetic data and other operations...


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




