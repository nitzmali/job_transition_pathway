import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
# Assuming you have the following data loaded into pandas DataFrames
# jobs_df = pd.DataFrame({'Job': ['Data Analyst', 'Data Scientist', ...], 'Skills': [['SQL', 'Python'], ['Machine Learning', 'Python', 'SQL'], ...]})
# courses_df = pd.DataFrame({'Course': ['Intro to Python', 'Advanced Machine Learning', ...], 'Skills_Taught': [['Python'], ['Machine Learning', 'Deep Learning'], ...]})
# skills_df = pd.DataFrame({'Skill': ['Python', 'SQL', 'Machine Learning', ...]})
# A simple function to select a random job and its required skills
def select_random_job(jobs_df):
    random_job = jobs_df.sample()
    return random_job.title_cleaned.values[0], random_job.skills_tagged.values[0]


def select_next_job(current_job, jobs_df, job_progression_dict):
    next_job,next_skills,random_job_selection_flag="","",0
    # If the current job is in the progression dict, select the next job from there
    if current_job in job_progression_dict:
        next_jobs = job_progression_dict[current_job]
        next_jobs=list(set(jobs_df["title_cleaned"].unique()).intersection(set(next_jobs)))
        if len(next_jobs)!=0:
            next_job = random.choice(next_jobs)
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
            next_job, next_skills = select_random_job(jobs_df)
            random_job_selection_flag=1
    else:
        next_job, next_skills = select_random_job(jobs_df)
        random_job_selection_flag=1
    return next_job, next_skills,random_job_selection_flag





# A simple function to select a course that teaches at least one of the required skills
def select_relevant_course(courses_df, required_skills):
    relevant_courses = courses_df[courses_df['Skills_Taught'].apply(lambda x: any(skill in x for skill in required_skills))]
    if not relevant_courses.empty:
        return relevant_courses.sample()
    return None


def calculate_course_score(course, skill_gap, weight_skill_match=1, weight_subscribers=0.001, weight_rating=0.5):
    # Calculate the proportion of skill gap covered by course
    skills_matched = sum(skill in course['skills_tagged'] for skill in skill_gap)
    skill_match_score = skills_matched / len(skill_gap) if skill_gap else 0

    # Factor in course popularity and rating
    #subscriber_score = course['num_subscribers'] * weight_subscribers if 'num_subscribers' in course else 0
    #rating_score = course['rating'] * weight_rating if 'rating' in course else 0

    # Calculate total score
    total_score = (skill_match_score * weight_skill_match) #+ subscriber_score + rating_score
    return total_score

def calculate_similarity_score(skill_gap,df_c_cleaned):

    def preprocess_text(text):
        return " ".join(token.lemma_.lower() for token in nlp(text) if not token.is_punct and not token.is_stop)

    skill_gap_text = " ".join(skill_gap)
    # Vectorize the text
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_c_cleaned["full_description"])
    skill_gap_vec = tfidf_vectorizer.transform([skill_gap_text])

    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(skill_gap_vec, tfidf_matrix).flatten()
    df_c_cleaned["cosine_score"]=cosine_similarities
    return df_c_cleaned

def select_relevant_course(df_c_cleaned, skill_gap):
    # Apply the scoring function to each course
    df_c_cleaned['course_score'] = df_c_cleaned.apply(calculate_course_score, axis=1, skill_gap=skill_gap)
    df_c_cleaned = calculate_similarity_score(skill_gap,df_c_cleaned)
    # Sort courses based on the calculated score
    #df_c_cleaned = df_c_cleaned.sort_values(by='course_score', ascending=False)
    df_c_cleaned = df_c_cleaned.sort_values(by='cosine_score', ascending=False)
    # Select the top course if available
    top_course = df_c_cleaned.iloc[0] if not df_c_cleaned.empty else None
    return top_course

# Select the most relevant course based on the skill gap
#relevant_course = select_relevant_course(df_c_cleaned, skill_gap)
#relevant_course.head()

# Generate synthetic user data
def generate_synthetic_data(jobs_df, courses_df, skills_df, num_samples=10):
    synthetic_data = []
    for _ in range(num_samples):
        #select_random_job(jobs_df)
        current_job, current_skills = select_random_job(jobs_df)
        # Select a job to progress to that requires at least one additional skill
        next_job, next_skills,random_job_selection_flag = select_next_job(current_job,jobs_df,sorted_full_job_progression_dict_lower)
        #while set(next_skills).issubset(current_skills):
        #    next_job, next_skills = select_random_job(jobs_df)
        next_skills=ast.literal_eval(next_skills)
        current_skills=ast.literal_eval(current_skills)
        # Find the skill gap
        skill_gap = list(set(next_skills) - set(current_skills))
        #print(skill_gap)
        # Find a course that teaches one of the skills in the gap
        course = select_relevant_course(courses_df, skill_gap)
        #print(course)
        if course is not None:
            course_name = course.course_title
            course_skills = course.skills_tagged
            course_title = course.title
        else:
            # If no single course covers the gap, select a random one for this synthetic example
            course_name = "Random Course"
            course_skills = "Random Skill"
        
        # Append to the synthetic data list
        synthetic_data.append({
            'Current Job': current_job,
            'Next Job': next_job,
            'Current Skills': current_skills,
            'Next Skills': next_skills,
            'Skill Gap': skill_gap,
            'course title':course_title,
            'Course Taken': course_name,
            'Course Skills': course_skills,
            'random next job':random_job_selection_flag
        })
    
    return pd.DataFrame(synthetic_data)


# Generate synthetic user data
def generate_synthetic_next_job_data(jobs_df,courses_df,synthetic_user_data):
    synthetic_data = []
    for i,iterrow in synthetic_user_data.iterrows():
        #print(i, iterrow["Next Job"],iterrow["Next Skills"])
        #select_random_job(jobs_df)
        current_job, current_skills = iterrow["Next Job"],(iterrow["Next Skills"])
        # Select a job to progress to that requires at least one additional skill
        next_job, next_skills,random_job_selection_flag = select_next_job(current_job,jobs_df,sorted_full_job_progression_dict_lower)
        #while set(next_skills).issubset(current_skills):
        #    next_job, next_skills = select_random_job(jobs_df)
        next_skills=ast.literal_eval(next_skills)
        # Find the skill gap
        skill_gap = list(set(next_skills) - set(current_skills))
        #print(skill_gap)
        # Find a course that teaches one of the skills in the gap
        course = select_relevant_course(courses_df, skill_gap)
        #print(course)
        if course is not None:
            course_name = course.course_title
            course_skills = course.skills_tagged
            course_title = course.title
        else:
            # If no single course covers the gap, select a random one for this synthetic example
            course_name = "Random Course"
            course_skills = "Random Skill"
        
        # Append to the synthetic data list
        synthetic_data.append({
            'Current Job': current_job,
            'Next Job': next_job,
            'Current Skills': current_skills,
            'Next Skills': next_skills,
            'Skill Gap': skill_gap,
            'course title':course_title,
            'Course Taken': course_name,
            'Course Skills': course_skills,
            'random next job':random_job_selection_flag
        })
    
    return pd.DataFrame(synthetic_data)



# Generate the synthetic data
synthetic_user_data_v1 = generate_synthetic_data(df_j_cleaned, df_c_cleaned, skills_list,10000)
synthetic_user__next_data_v2 = generate_synthetic_next_job_data(df_j_cleaned, df_c_cleaned,synthetic_user_data_v1)
synthetic_user__next_data_v3 = generate_synthetic_next_job_data(df_j_cleaned, df_c_cleaned,synthetic_user__next_data_v2)
synthetic_user__next_data_v4 = generate_synthetic_next_job_data(df_j_cleaned, df_c_cleaned,synthetic_user__next_data_v3)
synthetic_user_data = pd.concat([synthetic_user_data_v1,synthetic_user__next_data_v2,synthetic_user__next_data_v3,synthetic_user__next_data_v4],axis=0)
#print(synthetic_user_data.head())
