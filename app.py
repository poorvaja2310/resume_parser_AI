import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

from resume_parser import ResumeParser
from job_matcher import JobMatcher
from utils import setup_nltk_data, load_models

# Page config
st.set_page_config(
    page_title="AI Resume Parser & Job Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}

.resume-card {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    background: #f9f9f9;
}

.high-match {
    border-left: 5px solid #4CAF50;
}

.medium-match {
    border-left: 5px solid #FF9800;
}

.low-match {
    border-left: 5px solid #f44336;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_models():
    """Initialize and cache models"""
    return setup_nltk_data(), load_models()

def main():
    st.markdown('<h1 class="main-header">🤖 AI Resume Parser & Job Matcher</h1>', unsafe_allow_html=True)
    
    # Initialize models
    with st.spinner("Loading AI models... This may take a few moments on first run."):
        _, models = initialize_models()
    
    # Sidebar
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["Resume Parser", "Job Matching", "Analytics"])
    
    if page == "Resume Parser":
        resume_parser_page(models)
    elif page == "Job Matching":
        job_matching_page(models)
    elif page == "Analytics":
        analytics_page()

def resume_parser_page(models):
    st.header("📄 Resume Upload & Parsing")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Upload Resumes")
        uploaded_files = st.file_uploader(
            "Choose resume files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files"
        )
        
        parse_button = st.button("🚀 Parse Resumes", type="primary")
    
    with col2:
        if uploaded_files and parse_button:
            st.subheader("Parsing Results")
            
            parser = ResumeParser(models)
            parsed_resumes = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Parsing {uploaded_file.name}...")
                
                try:
                    # Parse resume
                    resume_data = parser.parse_resume(uploaded_file)
                    parsed_resumes.append({
                        'filename': uploaded_file.name,
                        'data': resume_data,
                        'status': 'success'
                    })
                    
                    # Display parsed resume
                    with st.expander(f"📋 {uploaded_file.name}", expanded=True):
                        display_parsed_resume(resume_data)
                    
                except Exception as e:
                    st.error(f"Error parsing {uploaded_file.name}: {str(e)}")
                    parsed_resumes.append({
                        'filename': uploaded_file.name,
                        'error': str(e),
                        'status': 'error'
                    })
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("Parsing complete!")
            
            # Store in session state
            st.session_state.parsed_resumes = parsed_resumes
            
            # Download JSON
            if parsed_resumes:
                json_data = json.dumps([r for r in parsed_resumes if r['status'] == 'success'], 
                                     indent=2, default=str)
                st.download_button(
                    "💾 Download Parsed Data (JSON)",
                    data=json_data,
                    file_name=f"parsed_resumes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

def job_matching_page(models):
    st.header("🎯 Job Matching & Ranking")
    
    if 'parsed_resumes' not in st.session_state or not st.session_state.parsed_resumes:
        st.warning("⚠️ Please parse some resumes first in the 'Resume Parser' page.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Job Description")
        job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer")
        job_description = st.text_area(
            "Job Description",
            height=300,
            placeholder="Paste the complete job description here..."
        )
        
        st.subheader("Matching Settings")
        weight_skills = st.slider("Skills Weight", 0.0, 1.0, 0.4)
        weight_experience = st.slider("Experience Weight", 0.0, 1.0, 0.3)
        weight_education = st.slider("Education Weight", 0.0, 1.0, 0.2)
        weight_overall = st.slider("Overall Fit Weight", 0.0, 1.0, 0.1)
        
        match_button = st.button("🔍 Match & Rank", type="primary")
    
    with col2:
        if job_description and match_button:
            st.subheader("Matching Results")
            
            job_matcher = JobMatcher(models)
            
            # Get successful parsed resumes
            successful_resumes = [r for r in st.session_state.parsed_resumes 
                                 if r['status'] == 'success']
            
            if not successful_resumes:
                st.error("No successfully parsed resumes found.")
                return
            
            with st.spinner("Analyzing job requirements and matching resumes..."):
                # Generate job benchmark
                job_benchmark = job_matcher.generate_job_benchmark(job_description, job_title)
                
                # Match and rank resumes
                weights = {
                    'skills': weight_skills,
                    'experience': weight_experience,
                    'education': weight_education,
                    'overall': weight_overall
                }
                
                ranked_resumes = job_matcher.rank_resumes(
                    successful_resumes, job_benchmark, weights
                )
            
            # Display job benchmark
            with st.expander("🎯 Generated Job Benchmark", expanded=False):
                display_job_benchmark(job_benchmark)
            
            # Display ranked results
            st.subheader("📊 Ranked Results")
            display_ranked_resumes(ranked_resumes)
            
            # Store results in session state
            st.session_state.ranked_resumes = ranked_resumes
            st.session_state.job_benchmark = job_benchmark

def analytics_page():
    st.header("📈 Analytics Dashboard")
    
    if 'ranked_resumes' not in st.session_state:
        st.warning("⚠️ Please complete job matching first to see analytics.")
        return
    
    ranked_resumes = st.session_state.ranked_resumes
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card"><h3>Total Resumes</h3><h2>{}</h2></div>'.format(
            len(ranked_resumes)), unsafe_allow_html=True)
    
    with col2:
        high_match = len([r for r in ranked_resumes if r['overall_score'] >= 0.7])
        st.markdown('<div class="metric-card"><h3>High Match</h3><h2>{}</h2></div>'.format(
            high_match), unsafe_allow_html=True)
    
    with col3:
        avg_score = sum(r['overall_score'] for r in ranked_resumes) / len(ranked_resumes)
        st.markdown('<div class="metric-card"><h3>Avg Score</h3><h2>{:.2f}</h2></div>'.format(
            avg_score), unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Score distribution
        scores = [r['overall_score'] for r in ranked_resumes]
        fig = px.histogram(x=scores, nbins=10, title="Score Distribution")
        fig.update_layout(xaxis_title="Overall Score", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Skills match vs Experience match
        skills_scores = [r['scores']['skills'] for r in ranked_resumes]
        exp_scores = [r['scores']['experience'] for r in ranked_resumes]
        
        fig = px.scatter(x=skills_scores, y=exp_scores, 
                        title="Skills vs Experience Scores")
        fig.update_layout(xaxis_title="Skills Score", yaxis_title="Experience Score")
        st.plotly_chart(fig, use_container_width=True)
    
    # Skills word cloud
    if st.checkbox("Show Skills Word Cloud"):
        all_skills = []
        for resume in ranked_resumes:
            all_skills.extend(resume['resume_data']['skills'])
        
        if all_skills:
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white').generate(' '.join(all_skills))
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

def display_parsed_resume(resume_data):
    """Display parsed resume data in a structured format"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**👤 Personal Info:**")
        st.write(f"• Name: {resume_data.get('name', 'Not found')}")
        st.write(f"• Email: {resume_data.get('email', 'Not found')}")
        st.write(f"• Phone: {resume_data.get('phone', 'Not found')}")
        
        st.write("**🎓 Education:**")
        education = resume_data.get('education', [])
        if education:
            for edu in education[:3]:  # Show top 3
                st.write(f"• {edu}")
        else:
            st.write("• Not found")
        
        st.write("**💼 Experience:**")
        st.write(f"• Years: {resume_data.get('experience_years', 'Not calculated')}")
    
    with col2:
        st.write("**🛠️ Skills:**")
        skills = resume_data.get('skills', [])
        if skills:
            skills_text = ", ".join(skills[:10])  # Show top 10 skills
            if len(skills) > 10:
                skills_text += f" ... (+{len(skills)-10} more)"
            st.write(skills_text)
        else:
            st.write("Not found")
        
        st.write("**🏢 Companies:**")
        companies = resume_data.get('companies', [])
        if companies:
            for company in companies[:5]:  # Show top 5
                st.write(f"• {company}")
        else:
            st.write("• Not found")

def display_job_benchmark(benchmark):
    """Display generated job benchmark"""
    st.json(benchmark)

def display_ranked_resumes(ranked_resumes):
    """Display ranked resumes with scores"""
    for i, resume in enumerate(ranked_resumes, 1):
        score = resume['overall_score']
        
        # Determine match level
        if score >= 0.7:
            match_class = "high-match"
            match_label = "🟢 High Match"
        elif score >= 0.5:
            match_class = "medium-match"
            match_label = "🟡 Medium Match"
        else:
            match_class = "low-match"
            match_label = "🔴 Low Match"
        
        with st.container():
            st.markdown(f'<div class="resume-card {match_class}">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**#{i} {resume['filename']}**")
                st.markdown(f"{match_label} - Overall Score: **{score:.2f}**")
            
            with col2:
                st.write("**Detailed Scores:**")
                for category, score_val in resume['scores'].items():
                    st.write(f"• {category.title()}: {score_val:.2f}")
            
            with col3:
                if st.button(f"View Details", key=f"detail_{i}"):
                    st.json(resume['resume_data'])
            
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()