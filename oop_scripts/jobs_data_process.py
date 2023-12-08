#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import re
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

class TextPreprocessor:
    def __init__(self):
        # Download necessary NLTK tokens
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')

        # Initialize the lemmatizer and stopwords
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Load the spaCy model
        self.nlp = spacy.load('en_core_web_md')

    def preprocess_text_spacy(self, text):
        doc = self.nlp(text)
        preprocessed_text = []
        for token in doc:
            if token.is_stop or token.is_punct or token.is_space:
                continue
            if token.lemma_ == 'datum':
                preprocessed_text.append('data')
            else:
                preprocessed_text.append(token.text.lower())
        return ' '.join(preprocessed_text)

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        words = nltk.word_tokenize(text)
        words = [word for word in words if word not in self.stop_words]
        return ' '.join(words)

    def clean_lem_stop(self, df, column_name):
        df[column_name] = df[column_name].apply(self.preprocess_text_spacy)
        df[column_name] = df[column_name].apply(self.clean_text)
        return df
    
# Usage
preprocessor = TextPreprocessor()


# In[2]:


import os
import pandas as pd
from os import listdir

class DataLoader:
    def __init__(self, job_path, courses_path, skills_path):
        self.job_path = job_path
        self.courses_path = courses_path
        self.skills_path = skills_path
        self.preprocessor = TextPreprocessor()

    def load_courses(self):
        files_c = [f for f in listdir(self.courses_path) if f.endswith(".csv")]
        df_c = pd.concat([pd.read_csv(os.path.join(self.courses_path, f)) for f in files_c])
        return df_c

    def load_skills(self):
        df_s = pd.read_csv(self.skills_path)
        return df_s

    def load_jobs(self):
        df_j_main_list = []
        job_dir = listdir(self.job_path)
        if '.DS_Store' in job_dir:
            job_dir.remove('.DS_Store')
        for job in job_dir:
            files_j = [f for f in listdir(os.path.join(self.job_path, job)) if f.endswith(".csv")]
            df_j = pd.concat([pd.read_csv(os.path.join(self.job_path, job, f)) for f in files_j])
            df_j["searched_title"] = job
            df_j_main_list.append(df_j)
        df_j = pd.concat(df_j_main_list)
        return df_j


    
    def skills_clean_load(self):
        ## clean skills dataset 
        #clean skills dataset 

        df_s = self.load_skills()
        skills = pd.DataFrame(df_s[df_s["Unnamed: 5"].notna()]["Unnamed: 5"].unique())
        skills.columns=["skills"]
        #skills["skills"] = skills["skills"].apply(preprocess_text_spacy)
        skills["skills"] = skills["skills"].apply(self.preprocessor.clean_text)
        skills_list = skills["skills"].tolist()
        skills_list.append('python')
        skills_list.append('python programming')
        skills_list.append('statistical')
        skills_list.append("r programming")

        for ach in ["docker",
         "neural network","matlab","google bard ai","ai governance","machine learning","tensorflow","computer vision","prompts","generate prompts","generative","generative ai","nlp","natural language processing","langchain",
         "pytorch","llm","scala","opencv"]:
            skills_list.append(ach)

            
        df_s_skills_list = pd.DataFrame(skills_list, columns=['skills_list'])
        return df_s_skills_list, skills_list
        
# Usage
'''
job_path = "/Users/gouribenni/Library/CloudStorage/GoogleDrive-gouri.benni@sjsu.edu/My Drive/Data Mining/data/raw/job_data"
courses_path = "/Users/gouribenni/Library/CloudStorage/GoogleDrive-gouri.benni@sjsu.edu/My Drive/Data Mining/data/raw/courses_data"
skills_path = "/Users/gouribenni/Library/CloudStorage/GoogleDrive-gouri.benni@sjsu.edu/My Drive/Data Mining/data/processed_datasets/skills_dataset/skills_df_updated.csv"
'''




# In[3]:


import os
import pandas as pd
from os import listdir

class DataWriter:
    def __init__(self):
        pass
    def save_data_frame(self,df,path):
        df.to_csv(path)
    def save_json(self,json_f,path):
        with open(path, 'w') as json_file:
            json.dump(json_f, json_file)
    


# In[ ]:





# In[4]:


## clean jobs dataset 
import ast 
import numpy as np
class clean_jobs_data:
    def __init__(self, df):
        self.df = df
        self.preprocessor = TextPreprocessor()
        
    def clean_job_df(self, df):
        df = self.preprocessor.clean_lem_stop(df,"title")
        df = self.preprocessor.clean_lem_stop(df,"jobDescription")
        job_descriptions = df_j["jobDescription"].tolist()
        return df,job_descriptions

    def sdnjsj():
        columns_of_interest = ['company',
         'companyRating',
         'companyReviewCount',
         'displayTitle',
         'employerAssistEnabled',
         'employerResponsive',
         'extractedSalary',
         'featuredEmployer',
         'featuredEmployerCandidate',
         'formattedLocation',
         'formattedRelativeTime',
         'highVolumeHiringModel',
         'hiringEventJob',
         'indeedApplyEnabled',
         'indeedApplyable',
         'isJobVisited',
         'isMobileThirdPartyApplyable',
         'isNoResumeJob',
         'isSubsidiaryJob',
         'jobCardRequirementsModel',
         'jobLocationCity',
         'jobLocationState',
         'locationCount',
         'newJob',
         'normTitle',
         'openInterviewsInterviewsOnTheSpot',
         'openInterviewsJob',
         'openInterviewsOffersOnTheSpot',
         'openInterviewsPhoneJob',
         'overrideIndeedApplyText',
         'remoteLocation',
         'resumeMatch',
         'salarySnippet',
         'showAttainabilityBadge',
         'showCommutePromo',
         'showEarlyApply',
         'showJobType',
         'showRelativeDate',
         'showSponsoredLabel',
         'showStrongerAppliedLabel',
         'smartFillEnabled',
         'smbD2iEnabled',
         'snippet',
         'sponsored',
         'title',
         'truncatedCompany',
         'urgentlyHiring',
         'vjFeaturedEmployerCandidate',
         'jobDescription','searched_title']

    def salary_snippet_to_dict(self,row):
        if pd.isna(row):
            # Return a dictionary with NaN values if the row is NaN
            return {'currency': np.nan, 'salaryTextFormatted': np.nan, 'source': np.nan,'text':np.nan}
        elif isinstance(row, dict):
            # Return the row as is if it's already a dictionary
            return row
        else:
            # If the row is a string representation of a dictionary (assumed if not NaN or dict),
            # safely evaluate it to a dictionary here
            # Note: Be cautious with `eval`. Here it's mentioned for potential string to dict conversion.
            # In a secure context, confirm the string format and consider `ast.literal_eval` instead.
            try:
                # Convert string to dictionary safely
                dict_row = eval(row)
                return dict_row if isinstance(dict_row, dict) else {'max': np.nan, 'min': np.nan, 'type': np.nan}
            except:
                # In case of error during eval, return NaN values
                return {'currency': np.nan, 'salaryTextFormatted': np.nan, 'source': np.nan,'text':np.nan}

    def salary_to_dict(self,row):
        if pd.isna(row):
            # Return a dictionary with NaN values if the row is NaN
            return {'max': np.nan, 'min': np.nan, 'type': np.nan}
        elif isinstance(row, dict):
            # Return the row as is if it's already a dictionary
            return row
        else:
            # If the row is a string representation of a dictionary (assumed if not NaN or dict),
            # safely evaluate it to a dictionary here
            # Note: Be cautious with `eval`. Here it's mentioned for potential string to dict conversion.
            # In a secure context, confirm the string format and consider `ast.literal_eval` instead.
            try:
                # Convert string to dictionary safely
                dict_row = eval(row)
                return dict_row if isinstance(dict_row, dict) else {'max': np.nan, 'min': np.nan, 'type': np.nan}
            except:
                # In case of error during eval, return NaN values
                return {'max': np.nan, 'min': np.nan, 'type': np.nan}





# In[ ]:





# In[5]:


import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.util import filter_spans

class phrase_matcher:
    def __init__(self):
        pass 

    def phrase_matcher_model(self,description,skills_list):
        nlp = spacy.load("en_core_web_md")  # Load the model
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")  # Create the matcher object
    
        # Assuming 'skills_list' is a list of skills, and 'job_descriptions' is a list containing job descriptions
    
        # Add patterns to the matcher. Patterns are made by converting each skill string into a Doc object
        patterns = [nlp.make_doc(skill) for skill in skills_list]
        matcher.add("Skills", patterns)
    
        # Process the job description to create a Spacy Doc
        doc = nlp(description)
    
        # Match the patterns to the doc
        matches = matcher(doc)
    
        # Create Span objects for the matched sequences
        spans = [Span(doc, start, end, label="SKILL") for match_id, start, end in matches]
    
        # Filter the spans to remove overlaps
        filtered_spans = filter_spans(spans)
    
        # Now you can create new entities in the doc using the filtered spans
        doc.ents = filtered_spans  # Overwrite or append to doc.ents with the non-overlapping skill entities
        entities_extracted = []
        # Print the entities in the document
        for ent in doc.ents:
            entities_extracted.append(ent.text)
        '''
        for each in matches:
            #print(each)
            if each.lower_ == "statistical":
                for skills_identify in skills_list:
                    if each.lower_ in skills_identify:
                        #print(skills_identify)
                        entities_extracted.append(skills_identify)
        '''
        return set(entities_extracted)


# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


class define_job_progression_dictionary:
    def __init__(self):
        pass 

    def define_job_progression(self):

            Job_progression_dictionary =  {
            "Junior Data Analyst" : ["Data Analyst", "Senior Data Analyst", "Data Engineer", "Senior Data Engineer", "Lead Data Engineer", "Data Architect", "Data Manager", "Director of Analytics", "Chief Data Officer"],
            "Junior Data Scientist" : ["Data Scientist", "Senior Data Scientist", "Lead Data Scientist", "Principal Data Scientist", "Data Science Manager", "Director of Data Science", "Vice President of Data Science", "Chief Data Scientist"],
            "Junior Data Engineer" : ["Data Engineer", "Senior Data Engineer", "Data Pipeline Architect", "Data Engineering Manager", "Director of Data Engineering", "Chief Technology Officer (CTO)"],
            "Business Intelligence Analyst" : ["Senior Business Intelligence Analyst", "Business Intelligence Manager", "Director of Business Intelligence", "Vice President of Business Intelligence", "Chief Intelligence Officer"],
            "Machine Learning Engineer" : ["Senior Machine Learning Engineer", "Machine Learning Architect", "Machine Learning Manager", "Head of Machine Learning", "Chief AI Officer"],
            "Data Analyst" : ["Data Engineer", "Senior Data Engineer", "Data Architect", "Senior Data Architect", "Enterprise Architect", "Chief Technology Officer (CTO)"],
            "Data Analyst" : ["Big Data Engineer", "Senior Big Data Engineer", "Big Data Architect", "Head of Big Data", "Director of Data Engineering", "Chief Information Officer (CIO)"],
            "Junior AI Developer" : ["AI Developer", "Senior AI Developer", "AI Architect", "AI Project Manager", "Head of AI", "Chief AI Officer"],
            "Junior DBA" : ["Mid-level DBA", "Senior DBA", "Database Manager", "Data Architect", "Director of Database Management", "Chief Information Officer"],
            "Junior Statistician" : ["Statistician" , "Senior Statistician", "Quantitative Analyst", "Senior Quantitative Analyst", "Quantitative Research Manager", "Director of Quantitative Research"],
            "Junior Business Analyst" : ["Business Analyst", "Senior Business Analyst", "Business Analysis Manager", "Business Intelligence Analyst", "Director of Business Analysis", "Chief Strategy Officer"],
            "Research Assistant" : ["Research Analyst", "Research Scientist", "Senior Research Scientist", "Principal Scientist", "Director of Research", "Chief Science Officer"],
            "Data Analyst" : ["Product Analyst", "Data Product Manager", "Senior Data Product Manager", "Director of Product Management", "Vice President of Product", "Chief Product Officer"]
            }

            original_dict = {
                "Junior Data Analyst": ["Data Analyst", "Senior Data Analyst", "Data Engineer", "Senior Data Engineer", 
                                        "Lead Data Engineer", "Data Architect", "Data Manager", "Director of Analytics", 
                                        "Chief Data Officer"],
                "Junior Data Scientist": ["Data Scientist", "Senior Data Scientist", "Lead Data Scientist", 
                                          "Principal Data Scientist", "Data Science Manager", "Director of Data Science", 
                                          "Vice President of Data Science", "Chief Data Scientist"],
                "Junior Data Engineer": ["Data Engineer", "Senior Data Engineer", "Data Pipeline Architect", 
                                         "Data Engineering Manager", "Director of Data Engineering", "Chief Technology Officer"],
                "Business Intelligence Analyst": ["Senior Business Intelligence Analyst", "Business Intelligence Manager", 
                                                  "Director of Business Intelligence", "Vice President of Business Intelligence", 
                                                  "Chief Intelligence Officer"],
                "Machine Learning Engineer": ["Senior Machine Learning Engineer", "Machine Learning Architect", 
                                              "Machine Learning Manager", "Head of Machine Learning", "Chief AI Officer"],
                "Junior AI Developer": ["AI Developer", "Senior AI Developer", "AI Architect", "AI Project Manager", 
                                        "Head of AI", "Chief AI Officer"],
                "Junior DataBase Administrator": ["DataBase Administrator", "Senior DataBase Administrator", "Database Manager", "Data Architect", "Director of Database Management", 
                               "Chief Information Officer"],
                "Junior Statistician": ["Statistician", "Senior Statistician", "Quantitative Analyst", "Senior Quantitative Analyst", 
                                        "Quantitative Research Manager", "Director of Quantitative Research"],
                "Junior Business Analyst": ["Business Analyst", "Senior Business Analyst", "Business Analysis Manager", 
                                            "Business Intelligence Analyst", "Director of Business Analysis", "Chief Strategy Officer"],
                "Research Assistant": ["Research Analyst", "Research Scientist", "Senior Research Scientist", "Principal Scientist", 
                                       "Director of Research", "Chief Science Officer"],
                # Multiple entries for Data Analyst have been combined to include all unique progressions
                "Data Analyst": ["Data Engineer", "Senior Data Engineer", "Lead Data Engineer", "Data Architect", 
                                 "Senior Data Architect", "Enterprise Architect", "Product Analyst", "Data Product Manager", 
                                 "Senior Data Product Manager", "Director of Product Management", "Vice President of Product", 
                                 "Chief Product Officer", "Big Data Engineer", "Senior Big Data Engineer", "Big Data Architect", 
                                 "Head of Big Data", "Director of Data Engineering", "Chief Information Officer"]
            }

            # Define lateral moves for the given roles
            lateral_moves = {
                "Data Analyst": ["Business Intelligence Analyst", "Machine Learning Engineer"],
                "Data Scientist": ["Data Engineer", "AI Developer"],
                "Data Engineer": ["Machine Learning Engineer", "Big Data Engineer"],
                "Business Intelligence Analyst": ["Data Analyst", "Data Scientist"],
                "Machine Learning Engineer": ["Data Scientist", "AI Developer"],
                "AI Developer": ["Machine Learning Engineer", "Data Engineer"],
                "Data Base Administrator": ["Data Engineer", "Data Analyst"],
                "Statistician": ["Data Analyst", "Data Scientist"],
                "Business Analyst": ["Data Analyst", "Business Intelligence Analyst"],
                "Research Analyst": ["Data Scientist", "Statistician"]
            }
            return Job_progression_dictionary,original_dict,lateral_moves

    
    # Since we want to include lateral moves for each value in the original dictionary, 
    # we will create a function that merges the direct progressions and lateral moves into one list.
    
    # Function to merge progression and lateral moves
    def merge_progression_and_lateral_moves(self,direct_progression, lateral_move_titles):
        # Start with direct progression
        full_progression = direct_progression.copy()
        
        # Add lateral moves for each title in the direct progression if they exist
        for title in direct_progression:
            lateral_titles = lateral_moves.get(title, [])
            for lateral_title in lateral_titles:
                if lateral_title not in full_progression:  # Avoid duplicates
                    full_progression.append(lateral_title)
        
        return full_progression
    
    # Function to build a full job progression dictionary for each title
    def build_full_progression_dict(self,original_dict, lateral_moves):
        full_progression_dict = {}
        
        # Iterate over each starting job title
        for start_title, progression in original_dict.items():
            # Get the full progression for the starting title
            full_progression = self.merge_progression_and_lateral_moves(progression, lateral_moves)
            
            # Add the full progression to the dictionary for the starting title
            full_progression_dict[start_title] = full_progression
            
            # Now iterate over each job within the progression to build their own progression paths
            for i, title in enumerate(progression):
                if title not in full_progression_dict:  # Only add if it doesn't already exist to avoid overwriting
                    # Get the progression for this title (which is the rest of the list after this title)
                    next_progression = self.merge_progression_and_lateral_moves(progression[i + 1:], lateral_moves)
                    full_progression_dict[title] = next_progression
        
        return full_progression_dict

            
        


# In[ ]:





# In[ ]:





# In[7]:


import re
import numpy as np
from fuzzywuzzy import process
class title_matcher:
    def __init__(self):
        pass

    # Function to normalize job titles
    def normalize_title(self,title):
        # Lowercase and remove non-alphanumeric characters, replace with spaces
        title = re.sub(r'[^a-z0-9]', ' ', title.lower())
        # Remove extra whitespace
        title = re.sub(r'\s+', ' ', title).strip()
    
            # Remove common prefixes/suffixes
        #title = re.sub(r'\b(senior|junior|associate|expert|technical|lead|l\d+)\b', '', title)
        title = re.sub(r'\b(sr|senior|expert|technical|lead|l\d+)\b', 'Senior', title)
        title = re.sub(r'\b(vp|vice president|l\d+)\b', 'Senior', title)
        title = re.sub(r'\b(III|6|5|4|3|l\d+)\b', 'Senior', title)
        # Replace specific terms with standardized equivalents
        title = re.sub(r'\b(sr\.?|senior)\b', 'senior', title)  # Replace 'sr' or 'senior' with 'senior'
        title = re.sub(r'\b(jr\.?|junior)\b', 'junior', title) 
    
        # Convert to lower case and remove special characters
        title = re.sub(r'[^a-z\s]', '', title.lower())
    
        # Strip extra whitespace
        title = re.sub(r'\s+', ' ', title).strip()
        return title
    # Function to find the best matching title from the base job titles
    def match_title_to_base(self,scraped_title, base_job_titles):
        # Normalize the scraped job title
        normalized_title = self.normalize_title(scraped_title)
        # Check if the normalized title exactly matches one of the base job titles
        if normalized_title in base_job_titles:
            return normalized_title  # Return the matching base title
        
        # Partial match checking - longer base titles are checked first to match more specific job titles
        sorted_base_titles = sorted(base_job_titles, key=len, reverse=True)
        for base_title in sorted_base_titles:
            if base_title in normalized_title:
                if "senior" in normalized_title and "senior" not in base_title:
                    return "senior "+base_title
                if "junior" in normalized_title and "junior" not in base_title:
                    return "junior "+base_title
                if "chief" in normalized_title and "chief" not in base_title:
                    return "chief "+base_title
                return base_title
        
        return "unmatched"  # Return "unmatched" or some default value if no match is found
    
    # Define a function to match job titles using fuzzy string matching
    def fuzzy_match_title(self,scraped_title, base_job_titles, threshold=90):
        # Use the process function to find the closest match above a certain score threshold
        best_match, score = process.extractOne(self.normalize_title(scraped_title), base_job_titles)
        # Only accept the match if the score is above the threshold
        return best_match if score >= threshold else "unmatched"
    
    


# In[ ]:





# In[ ]:





# In[8]:


from sklearn.preprocessing import StandardScaler
class clean_salary:
    def __init__(self):
        pass 
    def fill_na_type(self,min_sal,max_sal,type):
        if pd.isna(type):
            if min_sal//1000 or max_sal//1000:
                return "yearly"
            else:
                return "hourly"
        else:
            return type

    def adjust_max_salary(self,min_sal,type):
        if type=="yearly":
            return min_sal+10000
        else:
            return min_sal+10

    def clean_df_job(self,df_j_cleaned):
        for each in df_j_cleaned["title_cleaned"].unique():
            condition = df_j_cleaned["title_cleaned"]==each
            # Condition for filtering ColumnB
        
            # Calculate the average of ColumnB based on the condition
            average_value_min = df_j_cleaned.loc[condition, 'min'].mean()
            average_value_max = df_j_cleaned.loc[condition, 'max'].mean()
        
            # Fill missing values in ColumnA with this average
            df_j_cleaned.loc[condition,"min"] = df_j_cleaned.loc[condition,"min"].fillna(average_value_min)
            df_j_cleaned.loc[condition,"max"] = df_j_cleaned.loc[condition,"max"].fillna(average_value_max)
        
        # Assuming a standard work year of 40 hours/week and 52 weeks/year
        HOURS_PER_WEEK = 40
        WEEKS_PER_YEAR = 52
        condition = df_j_cleaned["type"]=="hourly"
        df_j_cleaned.loc[condition,"min"]=df_j_cleaned.loc[condition,"min"]*HOURS_PER_WEEK * WEEKS_PER_YEAR
        df_j_cleaned.loc[condition,"max"]=df_j_cleaned.loc[condition,"max"]*HOURS_PER_WEEK * WEEKS_PER_YEAR
        df_j_cleaned.loc[condition,"type"]="yearly"
        df_j_cleaned["min"].fillna(df_j_cleaned["min"].mean(),inplace=True)
        df_j_cleaned["max"].fillna(df_j_cleaned["max"].mean(),inplace=True)
        
        
        
        # Assuming df is your DataFrame and 'current_salary' is the salary column
        scaler = StandardScaler()
        df_j_cleaned['normalized_min'] = scaler.fit_transform(df_j_cleaned[['min']])
        df_j_cleaned['normalized_max'] = scaler.fit_transform(df_j_cleaned[['max']])
    
        return df_j_cleaned


# In[ ]:





# In[ ]:





# In[26]:





# In[27]:


class course_clean:
    def __init__(self):
        self.tp_cl = TextPreprocessor()
    def salary_to_dict(self,row):
        if pd.isna(row):
            # Return a dictionary with NaN values if the row is NaN
            return {'max': np.nan, 'min': np.nan, 'type': np.nan}
        elif isinstance(row, dict):
            # Return the row as is if it's already a dictionary
            return row
        else:
            # If the row is a string representation of a dictionary (assumed if not NaN or dict),
            # safely evaluate it to a dictionary here
            # Note: Be cautious with `eval`. Here it's mentioned for potential string to dict conversion.
            # In a secure context, confirm the string format and consider `ast.literal_eval` instead.
            try:
                # Convert string to dictionary safely
                dict_row = eval(row)
                return dict_row if isinstance(dict_row, dict) else {'max': np.nan, 'min': np.nan, 'type': np.nan}
            except:
                # In case of error during eval, return NaN values
                return {'max': np.nan, 'min': np.nan, 'type': np.nan}
    
    def  course_price_detail(self,x):
        if pd.isna(x):
            return np.nan
        else:
            cou_dict = ast.literal_eval(x)
            return cou_dict["amount"]
    
    def  category_primary(self,x):
        cou_dict = ast.literal_eval(x)
        return cou_dict["url"]
    
    def  labels_return_title(self,x):
        cou_dict = ast.literal_eval(x)
        if len(cou_dict)==0:
            return "UnDefined"
        return cou_dict[0]["title"]
    
    
    
    def  labels_return_url(self,x):
        cou_dict = ast.literal_eval(x)
        if len(cou_dict)==0:
            return "UnDefined"
        return cou_dict[0]["url"]
    
    
    def  labels_return_display_name(self,x):
        cou_dict = ast.literal_eval(x)
        if len(cou_dict)==0:
            return "UnDefined"
        return cou_dict[0]["display_name"]

    #clean course data 
    def clean_course_df(self,df):
        df = df.reset_index(drop=True)
        # Create an empty DataFrame with a column named 'single_space' that has one row with a space
        df_space = pd.DataFrame({"single_space": [" "] * len(df)})
        
        df["full_description"] = df["title"]+df_space["single_space"]+df["description"]+df_space["single_space"]+df["headline"]+df_space["single_space"]+df["what_you_will_learn_data"]+df_space["single_space"]+df["objectives"]
        df = self.tp_cl.clean_lem_stop(df,"full_description")
        course_description = df["full_description"].tolist()
        return df,course_description




# In[35]:


def main():

    job_path = "/Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/raw/job_data"
    courses_path = "/Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/raw/courses_data"
    skills_path = "/Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/skills_dataset/skills_df_updated.csv"
    
    data_loader = DataLoader(job_path, courses_path, skills_path)
    df_c = data_loader.load_courses()
    df_s = data_loader.load_skills()
    df_j = data_loader.load_jobs()
    df_j,df_s,df_c = df_j[:10],df_s[:10],df_c[:10]
    df_s_skills_list, skills_list = data_loader.skills_clean_load()
    
    clean_jobs = clean_jobs_data(df_j)
    df_j_cleaned, job_descriptions = clean_jobs.clean_job_df(df_j)
    
    # Apply the function to each row of the 'extractedSalary' column
    salary_dicts = df_j_cleaned['extractedSalary'].apply(clean_jobs.salary_to_dict)
    # Now that we have a series of dictionaries, use `json_normalize` to create a DataFrame
    salary_df = pd.json_normalize(salary_dicts)
    
    # Concatenate the new DataFrame with the original one
    df_j_cleaned = pd.concat([df_j_cleaned.drop('extractedSalary', axis=1).reset_index(), salary_df], axis=1)
    
    # Apply the function to each row of the 'extractedSalary' column
    salary_snippet_dicts = df_j_cleaned['salarySnippet'].apply(clean_jobs.salary_snippet_to_dict)
    # Now that we have a series of dictionaries, use `json_normalize` to create a DataFrame
    salary_snippet_df = pd.json_normalize(salary_snippet_dicts)
    
    # Concatenate the new DataFrame with the original one
    df_j_cleaned = pd.concat([df_j_cleaned.drop('salarySnippet', axis=1).reset_index(), salary_snippet_df], axis=1)
    
    pm = phrase_matcher()
    df_j_cleaned["skills_tagged"]=df_j_cleaned["jobDescription"].apply(lambda x:pm.phrase_matcher_model(x,skills_list))
    
    
    
    
    
    #df["extractedSalary"]=df["extractedSalary"].astype(str)
    #df["extractedSalary"]=df['extractedSalary'].apply(ast.literal_eval)
    
    job_progress_cl = define_job_progression_dictionary()
    Job_progression_dictionary,original_dict,lateral_moves = job_progress_cl.define_job_progression()
    
    
    
    # Building the full job progression dictionary
    full_job_progression_dict = job_progress_cl.build_full_progression_dict(original_dict, lateral_moves)
    
    # Sorting the dictionary for better readability
    sorted_full_job_progression_dict = {k: full_job_progression_dict[k] for k in sorted(full_job_progression_dict)}
    
    
    unique_job_title_full = set()
    for key,value in Job_progression_dictionary.items():
        unique_job_title_full.add(key)
        for each in value:
            unique_job_title_full.add(each)
    
    
    unique_job_title = set()
    for key,value in sorted_full_job_progression_dict.items():
        unique_job_title.add(key)
        for each in value:
            unique_job_title.add(each)
    
    unique_job_title.difference(unique_job_title_full)
    
    sorted_full_job_progression_dict
    
    sorted_full_job_progression_dict_lower={}
    for k,v in sorted_full_job_progression_dict.items():
        sorted_full_job_progression_dict_lower[k.lower()]=[each.lower() for each in v]
    
    sorted_full_job_progression_dict_lower
    
    title_matcher_cl =title_matcher()

    # Create a set of base job titles from your job progression dictionary
    base_job_titles = set()
    for titles_list in sorted_full_job_progression_dict.values():
        for title in titles_list:
            base_job_titles.add(title_matcher_cl.normalize_title(title))  # Add the normalized base title
    
    df_j_cleaned["title_cleaned"] = df_j_cleaned["title"].apply(lambda x: title_matcher_cl.fuzzy_match_title(x,base_job_titles))
    df_j_cleaned["title_cleaned"] = df_j_cleaned["title_cleaned"].apply(lambda x: title_matcher_cl.match_title_to_base(x,base_job_titles))
    df_j_cleaned["title_cleaned"] = df_j_cleaned.apply(
        lambda x: x["title_cleaned"] if x["title_cleaned"] != "unmatched" else title_matcher_cl.fuzzy_match_title(x["normTitle"], base_job_titles),
        axis=1
    )
    df_j_cleaned["title_cleaned"] = df_j_cleaned.apply(
        lambda x: x["title_cleaned"] if x["title_cleaned"] != "unmatched" else title_matcher_cl.match_title_to_base(x["normTitle"], base_job_titles),
        axis=1
    )
    #df_j_cleaned["displayTitle"] = df_j_cleaned["displayTitle"].apply(lambda x: fuzzy_match_title(x,base_job_titles))
    #df_j_cleaned["displayTitle"] = df_j_cleaned["displayTitle"].apply(lambda x: match_title_to_base(x,base_job_titles))
    
    #df_j_cleaned["title"] = df_j_cleaned["title"].apply(lambda x: match_title_to_base(x,base_job_titles))
    
    columns_most_important = ['company', 'truncatedCompany',
     'companyRating',
     'companyReviewCount',
      'searched_title',
        'title',
     'title_cleaned',
     'normTitle',
     'min',
     'max',
     'type',
     'snippet',
     'jobDescription',
     'skills_tagged']
    
    df_j_cleaned = df_j_cleaned[columns_most_important]
    
    condition =df_j_cleaned["title_cleaned"]=="unmatched"
    df_j_cleaned.loc[condition,"title_cleaned"]=df_j_cleaned.loc[condition,"searched_title"]
    df_j_cleaned["title_cleaned"] = df_j_cleaned["title_cleaned"].apply(lambda x:x.lower()) 
    
    df_j_cleaned["title_cleaned"] = df_j_cleaned.apply(
        lambda x: x["title_cleaned"] if x["title_cleaned"] != "unmatched" else fuzzy_match_title(x["searched_title"], base_job_titles),
        axis=1
    )
    
    df_j_cleaned["title_cleaned"]=df_j_cleaned["title_cleaned"].apply(lambda x:x.replace("(cto)","").strip())
    df_j_cleaned["title_cleaned"]=df_j_cleaned["title_cleaned"].apply(lambda x:x.replace("(cio)","").strip())
    df_j_cleaned["title_cleaned"].unique()
    
    
    clean_salary_cl =clean_salary()
    df_j_cleaned["type"]=df_j_cleaned.apply(lambda x: clean_salary_cl.fill_na_type(x["min"],x["max"],x["type"]),axis=1)
    df_j_cleaned["max"]=df_j_cleaned[["min","type"]].apply(lambda x :clean_salary_cl.adjust_max_salary(x["min"],x["type"]),axis=1)
    df_j_cleaned = clean_salary_cl.clean_df_job(df_j_cleaned)

    ## courses clean
    course_clean_cl =course_clean()
    df_c_cleaned = df_c
    df_c_cleaned.columns.tolist()
    
    ## Lemmitization and cleaning columns
    
    df_c_cleaned["price"] = df_c_cleaned["price_detail"].apply(lambda x: course_clean_cl.course_price_detail(x))
    df_c_cleaned["requirements_data"] = df_c_cleaned["requirements_data"].apply(lambda x:ast.literal_eval(x)[0])
    df_c_cleaned["course_title"]=df_c_cleaned["labels"].apply(lambda x:course_clean_cl.labels_return_title(x))
    df_c_cleaned["course_url"] =df_c_cleaned["labels"].apply(lambda x:course_clean_cl.labels_return_url(x))
    df_c_cleaned["course_display_name"] =df_c_cleaned["labels"].apply(lambda x:course_clean_cl.labels_return_display_name(x))
    df_c_cleaned["objectives"] =df_c_cleaned["objectives"].apply(lambda x:ast.literal_eval(x)[0])
    del df_c_cleaned["labels"]
    df_c_cleaned["what_you_will_learn_data"] = df_c_cleaned["what_you_will_learn_data"].apply(lambda x:ast.literal_eval(x)[0])
    df_c_cleaned["target_audiences"] = df_c_cleaned["target_audiences"].apply(lambda x:ast.literal_eval(x)[0])
    del df_c_cleaned["price_detail"]
    
    ## merge all description into one    
    
    
    clean_course_columns =["title",
    "description",
    "headline",
    "requirements_data",
    "what_you_will_learn_data",
    "target_audiences",
    "objectives",
    #"full_description",
    "course_title",
    "course_display_name"]
    
    tp_cl = TextPreprocessor()
    for clean_column in clean_course_columns:
        df_c_cleaned[clean_column]=df_c_cleaned[clean_column].astype(str)
        df_c_cleaned = tp_cl.clean_lem_stop(df_c_cleaned,clean_column)
    
    df_c_cleaned,course_descriptions = course_clean_cl.clean_course_df(df_c_cleaned)
    
    
    
    df_c_cleaned_2 = df_c_cleaned.copy()
    df_c_cleaned_2["skills_tagged"] = df_c_cleaned_2["full_description"].apply(lambda x:pm.phrase_matcher_model(x,skills_list))
    
    course_columns_imp = [
     'title',
     'url',
     'description',
     'headline',
     'num_subscribers',
     'rating',
     'num_reviews',
     'num_quizzes',
     'num_lectures',
     'num_curriculum_items',
     'requirements_data',
     'what_you_will_learn_data',
     'target_audiences',
     'estimated_content_length',
     'content_info',
     'instructional_level',
     'objectives',
     'full_description',
     'skills_tagged']
    
    df_c_cleaned_2 = df_c_cleaned_2[course_columns_imp]

    ## save dataframes
    '''
    dw = DataWriter()
    dw.save_data_frame(df_c_cleaned_2,"/Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/course_cleaned.csv")

    dw.save_data_frame(df_j_cleaned,"/Users/nyzy/Library/CloudStorage/GoogleDrive-nitinmali999@gmail.com/.shortcut-targets-by-id/1yEbTjc1DwwTDd2CB86Il6F-3DhFUn8pU/Data Mining/data/processed_datasets/job_cleaned_all_titles_salary.csv"")
    '''




# In[34]:


# Call the main function
if __name__ == "__main__":
    main()


# In[29]:





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




