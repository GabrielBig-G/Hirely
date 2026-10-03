import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# --- Load saved artifacts ---
rf = joblib.load('cv_screening_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
feature_cols = joblib.load('feature_cols.pkl')
field_cols = joblib.load('field_cols.pkl')

# --- Helper functions (same as notebook) ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s,]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def skills_to_list(skill_string):
    return [s.strip().lower() for s in str(skill_string).split(',') if s.strip()]

def get_degree_level(edu):
    edu = edu.lower()
    if 'phd' in edu:
        return 3
    elif 'm.sc' in edu or 'm.eng' in edu:
        return 2
    elif 'b.sc' in edu or 'b.eng' in edu:
        return 1
    return 0

def skill_overlap_features(candidate_skills, required_skills, preferred_skills):
    cand_set = set(candidate_skills)
    req_set = set(required_skills)
    pref_set = set(preferred_skills)
    matched_required = cand_set & req_set
    matched_preferred = cand_set & pref_set
    req_coverage = len(matched_required) / len(req_set) if req_set else 0
    pref_coverage = len(matched_preferred) / len(pref_set) if pref_set else 0
    return {
        'my_required_coverage': req_coverage,
        'my_preferred_coverage': pref_coverage,
        'num_required_matched': len(matched_required),
        'num_required_missing': len(req_set - cand_set),
        'num_extra_skills': len(cand_set - req_set - pref_set)
    }

def match_candidate(candidate_skills_str, candidate_experience_years, candidate_projects_count,
                     candidate_education, candidate_work_auth, candidate_salary_expectation,
                     resume_summary_text, required_skills_str, preferred_skills_str,
                     min_experience_years, salary_min, salary_max, job_description_text):

    cand_skills = skills_to_list(candidate_skills_str)
    req_skills = skills_to_list(required_skills_str)
    pref_skills = skills_to_list(preferred_skills_str)

    resume_clean = clean_text(resume_summary_text)
    job_clean = clean_text(job_description_text)

    degree_lvl = get_degree_level(candidate_education)
    field = re.sub(r'^(B\.Sc|B\.Eng|M\.Sc|PhD)\s+', '', candidate_education)
    work_auth_encoded = 1 if candidate_work_auth == 'Authorized' else 0

    skill_feats = skill_overlap_features(cand_skills, req_skills, pref_skills)

    resume_vec = vectorizer.transform([resume_clean])
    job_vec = vectorizer.transform([job_clean])
    text_sim = cosine_similarity(resume_vec, job_vec)[0][0]

    exp_gap = candidate_experience_years - min_experience_years
    sal_gap = candidate_salary_expectation - salary_max

    row = {
        'experience_years': candidate_experience_years,
        'projects_count': candidate_projects_count,
        'salary_expectation_usd': candidate_salary_expectation,
        'minimum_experience_years': min_experience_years,
        'salary_min_usd': salary_min,
        'salary_max_usd': salary_max,
        'degree_level': degree_lvl,
        'work_authorization_encoded': work_auth_encoded,
        'my_required_coverage': skill_feats['my_required_coverage'],
        'my_preferred_coverage': skill_feats['my_preferred_coverage'],
        'num_required_matched': skill_feats['num_required_matched'],
        'num_required_missing': skill_feats['num_required_missing'],
        'num_extra_skills': skill_feats['num_extra_skills'],
        'text_similarity': text_sim,
        'experience_gap': exp_gap,
        'salary_gap': sal_gap,
    }
    for col in field_cols:
        row[col] = 1 if col == f'field_{field}' else 0

    X_new = pd.DataFrame([row])[feature_cols]
    prediction = rf.predict(X_new)[0]
    probabilities = rf.predict_proba(X_new)[0]
    prob_dict = dict(zip(rf.classes_, probabilities))
    confidence = max(probabilities) * 100

    matched_required = set(cand_skills) & set(req_skills)
    missing_required = set(req_skills) - set(cand_skills)
    exp_relevance = "HIGH" if exp_gap >= 0 else ("MEDIUM" if exp_gap >= -2 else "LOW")

    return {
        'prediction': prediction,
        'confidence': confidence,
        'prob_dict': prob_dict,
        'matched_required': matched_required,
        'missing_required': missing_required,
        'exp_relevance': exp_relevance
    }

# --- Streamlit UI ---
st.set_page_config(page_title="AI CV Screening System", layout="centered")
st.title("🤖 Hirely")
st.caption("An assistive screening tool that helps in hiring decision making...")

st.subheader("Job Description")
job_title = st.text_input("Job Title", "Backend Developer")
required_skills_str = st.text_input("Required Skills (comma-separated)", "Python, Django, REST API, PostgreSQL, Git, React")
preferred_skills_str = st.text_input("Preferred Skills (comma-separated)", "Docker, AWS")
min_experience_years = st.number_input("Minimum Experience (years)", min_value=0, value=2)
col1, col2 = st.columns(2)
salary_min = col1.number_input("Salary Min (USD)", min_value=0, value=70000)
salary_max = col2.number_input("Salary Max (USD)", min_value=0, value=110000)
job_description_text = st.text_area("Job Description Text",
    "We are looking for a backend developer with strong Python and Django experience, PostgreSQL, Git, and React knowledge, building REST APIs.")

st.subheader("Candidate CV")
candidate_skills_str = st.text_input("Candidate Skills (comma-separated)", "Python, Django, PostgreSQL, Git, JavaScript, React")
candidate_experience_years = st.number_input("Candidate Experience (years)", min_value=0, value=3)
candidate_projects_count = st.number_input("Projects Count", min_value=0, value=4)
candidate_education = st.selectbox("Education", [
    'B.Sc Computer Science', 'B.Sc Information Technology', 'M.Sc Data Science',
    'M.Sc Data Engineering', 'B.Eng Software Engineering', 'PhD Statistics',
    'PhD Computer Science', 'B.Sc Statistics', 'B.Sc Information Systems',
    'M.Sc Cloud Computing', 'M.Sc Artificial Intelligence', 'M.Sc Cybersecurity',
    'M.Sc Machine Learning', 'B.Eng Information Technology', 'M.Sc Computer Science',
    'B.Sc Cybersecurity'
])
candidate_work_auth = st.selectbox("Work Authorization", ["Authorized", "Requires sponsorship"])
candidate_salary_expectation = st.number_input("Salary Expectation (USD)", min_value=0, value=90000)
resume_summary_text = st.text_area("Resume Summary",
    "Experienced backend developer skilled in Python and Django, with growing frontend experience in React.")

if st.button("Screen Candidate"):
    result = match_candidate(
        candidate_skills_str, candidate_experience_years, candidate_projects_count,
        candidate_education, candidate_work_auth, candidate_salary_expectation,
        resume_summary_text, required_skills_str, preferred_skills_str,
        min_experience_years, salary_min, salary_max, job_description_text
    )

    st.divider()
    st.subheader("Result")
    rec = result['prediction']
    color = {"Interview": "green", "Manual Review": "orange", "Not Qualified": "red"}[rec]
    st.markdown(f"### Recommendation: :{color}[{rec.upper()}]")
    st.progress(int(result['confidence']))
    st.write(f"Confidence: {result['confidence']:.0f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Matched Required Skills**")
        for s in result['matched_required']:
            st.write(f"✅ {s}")
    with col2:
        st.write("**Missing Required Skills**")
        if result['missing_required']:
            for s in result['missing_required']:
                st.write(f"❌ {s}")
        else:
            st.write("None")

    st.write(f"**Experience Relevance:** {result['exp_relevance']}")

    st.write("**Class Probabilities**")
    st.bar_chart(pd.Series(result['prob_dict']))

    st.info("⚠️ This is an assistive recommendation only. Final hiring decisions should always involve human review.")