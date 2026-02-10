import streamlit as st
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import google.generativeai as genai
import tempfile

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Analyzer (Recruiter-Grade ATS + Gemini AI)")
st.caption("ATS • Skill Gaps • Resume Rewrite • Gemini AI • Ranking")

# =========================
# SKILL DATABASE
# =========================
ALL_SKILLS = [
    "python","java","sql","machine learning","deep learning","data analysis",
    "nlp","tensorflow","pytorch","html","css","javascript","react",
    "flask","streamlit","git","excel","statistics"
]

skill_recommendations = {
    "python": "Practice DSA + build automation & ML projects",
    "sql": "Master joins, subqueries, indexing",
    "machine learning": "Build regression & classification models",
    "deep learning": "Learn CNNs & backpropagation",
    "nlp": "Work with TF-IDF & transformers",
}

# =========================
# FUNCTIONS
# =========================
def extract_text_from_pdf(pdf):
    reader = PyPDF2.PdfReader(pdf)
    return " ".join(page.extract_text() for page in reader.pages).lower()

def extract_skills(text):
    return list(set(skill for skill in ALL_SKILLS if skill in text))

def job_match(resume, jd):
    vectors = CountVectorizer().fit_transform([resume, jd])
    sim = (vectors @ vectors.T).toarray()
    if sim[1][1] == 0:
        return 0.0
    return round((sim[0][1] / sim[1][1]) * 100, 2)

def resume_score(skills, match):
    return min(int(len(skills) * 5 + match), 100)

def ats_score(text, skills):
    score = 0
    if len(text.split()) > 300: score += 25
    if len(skills) >= 6: score += 25
    if "project" in text: score += 25
    if "experience" in text: score += 25
    return score

def generate_pdf(data):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp.name)
    styles = getSampleStyleSheet()
    content = [Paragraph(f"<b>{k}</b>: {v}", styles["Normal"]) for k, v in data.items()]
    doc.build(content)
    return temp.name

# =========================
# GEMINI AI FUNCTIONS
# =========================
def gemini_ai_analysis(api_key, resume_text, job_desc, role):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-pro")

    prompt = f"""
You are a senior technical recruiter.

Analyze the resume against the job role: {role}

Resume:
{resume_text}

Job Description:
{job_desc}

Provide:
1. ATS-style resume improvement bullets
2. Missing skills explanation
3. Recruiter verdict (Hire / Maybe / Reject)
4. Resume rewrite suggestions
5. Hiring probability percentage
"""

    response = model.generate_content(prompt)
    return response.text

# =========================
# INPUTS
# =========================
resume_file = st.file_uploader("📤 Upload Resume (PDF)", type=["pdf"])

jd_templates = {
    "Select role": "",
    "Software Engineer": "Python Java SQL Git DSA Projects Experience",
    "Data Analyst": "Python SQL Pandas Excel Statistics",
    "ML Engineer": "Python Machine Learning TensorFlow NLP Projects",
    "Web Developer": "HTML CSS JavaScript React Flask"
}

role = st.selectbox("🎯 Choose Job Role", jd_templates.keys())
job_desc = st.text_area("📋 Job Description", jd_templates[role], height=220)

st.subheader("🔑 Optional Gemini API")
api_key = st.text_input("Enter Gemini API Key (optional)", type="password")

# =========================
# ANALYSIS
# =========================
if st.button("🔍 Analyze Resume"):
    if resume_file and job_desc:
        resume_text = extract_text_from_pdf(resume_file)
        skills = extract_skills(resume_text)
        jd_skills = extract_skills(job_desc.lower())

        matched = list(set(skills) & set(jd_skills))
        missing = list(set(jd_skills) - set(skills))

        match = job_match(resume_text, job_desc)
        score = resume_score(skills, match)
        ats = ats_score(resume_text, skills)

        st.divider()
        st.subheader("📊 ATS Dashboard")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Resume Score", score)
        c2.metric("Job Match %", match)
        c3.metric("ATS Score", ats)
        c4.metric("Skills Found", len(skills))

        st.success(f"Matched Skills: {', '.join(matched) if matched else 'None'}")
        st.error(f"Missing Skills: {', '.join(missing) if missing else 'None'}")

        # =========================
        # GEMINI AI OUTPUT
        # =========================
        if api_key:
            st.divider()
            st.subheader("🤖 Gemini AI Recruiter Intelligence")

            with st.spinner("Gemini AI analyzing resume..."):
                ai_output = gemini_ai_analysis(
                    api_key,
                    resume_text,
                    job_desc,
                    role
                )
                st.markdown(ai_output)
        else:
            st.info("ℹ️ Add Gemini API key to enable AI resume rewriting & recruiter insights.")

        # =========================
        # PDF EXPORT
        # =========================
        pdf = generate_pdf({
            "Resume Score": score,
            "Job Match %": match,
            "ATS Score": ats,
            "Matched Skills": ", ".join(matched),
            "Missing Skills": ", ".join(missing)
        })

        with open(pdf, "rb") as f:
            st.download_button("📄 Download Resume Report", f, "Resume_Report.pdf")

    else:
        st.error("❌ Upload resume and job description")

# =========================
# MULTI-RESUME RANKING
# =========================
st.divider()
st.header("🏆 Resume Ranking Dashboard")

multi_files = st.file_uploader(
    "Upload Multiple Resumes",
    type=["pdf"],
    accept_multiple_files=True
)

if multi_files and job_desc:
    data = []
    for f in multi_files:
        text = extract_text_from_pdf(f)
        s = extract_skills(text)
        m = job_match(text, job_desc)
        data.append({
            "Candidate": f.name,
            "Score": resume_score(s, m),
            "Match %": m
        })

    df = pd.DataFrame(data).sort_values("Score", ascending=False)
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df.set_index("Candidate")["Score"])
