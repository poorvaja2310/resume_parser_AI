# AI Resume Parser & Job Matcher

A powerful AI-driven resume parser and job matching application built with Python and Streamlit. This application can parse resumes dynamically (like ChatGPT), extract comprehensive information, and rank candidates based on job requirements using advanced NLP models.

## Features

### 🔍 Advanced Resume Parsing
- **Multi-format Support**: PDF, DOCX, and TXT files
- **Dynamic Extraction**: No hardcoded templates - adapts to any resume format
- **Comprehensive Data**: Names, emails, phones, skills, experience, education, companies, projects, and certifications
- **AI-Powered**: Uses spaCy NLP and semantic analysis for intelligent extraction

### 🎯 Intelligent Job Matching
- **Dynamic Benchmark Generation**: Creates job requirements automatically from job descriptions
- **Semantic Analysis**: Uses MiniLM transformer models for deep understanding
- **Multi-dimensional Scoring**: Skills, experience, education, and overall fit analysis
- **Customizable Weights**: Adjust importance of different criteria
- **Detailed Match Reports**: Shows strengths, gaps, and recommendations

### 📊 Rich Analytics
- **Interactive Dashboards**: Score distributions, skill analysis, and trend visualization
- **Word Clouds**: Visual representation of candidate skills
- **Ranking System**: Automatic candidate ranking with match levels
- **Export Capabilities**: Download parsed data and results in JSON format

### 🚀 Modern Interface
- **Streamlit UI**: Clean, intuitive web interface
- **Real-time Processing**: Live parsing and matching with progress indicators
- **Responsive Design**: Works on desktop and mobile
- **Batch Processing**: Handle multiple resumes simultaneously

## Technology Stack

- **Python 3.8+**: Core programming language
- **Streamlit**: Web application framework
- **spaCy**: Advanced NLP processing
- **Sentence Transformers**: Semantic similarity using MiniLM
- **scikit-learn**: Machine learning utilities
- **PyPDF2 & pdfplumber**: PDF text extraction
- **python-docx**: DOCX document processing
- **NLTK**: Natural language toolkit
- **Plotly**: Interactive visualizations

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- At least 2GB RAM for model loading

### Quick Setup

1. **Clone or Download** the project files to a directory

2. **Install Dependencies**:
   ```bash
   py -m pip install -r requirements.txt
   ```

3. **Setup Models** (automated):
   ```bash
   py install_models.py
   ```
   
   This will automatically:
   - Install spaCy English model (`en_core_web_sm`)
   - Download required NLTK data
   - Verify all dependencies

4. **Start the Application**:
   ```bash
   py -m pip install streamlit
   py -m streamlit run app.py

   streamlit run app.py
   ```

### Manual Model Installation (if needed)
If the automated setup fails:
```bash
py -m spacy download en_core_web_sm
py -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```
py -m pip install --upgrade sentence-transformers
py -m pip install huggingface-hub==0.10.1
py -m pip install --upgrade sentence-transformers huggingface-hub transformers

py -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
py -m pip install --upgrade sentence-transformers transformers
py -c "import torch; print(torch.__version__)"
py -c "from sentence_transformers import SentenceTransformer; print(SentenceTransformer)"
## Usage Guide

### 1. Resume Parsing
1. Navigate to the "Resume Parser" tab
2. Upload one or multiple resume files (PDF, DOCX, TXT)
3. Click "🚀 Parse Resumes"
4. View extracted information for each resume
5. Download parsed data in JSON format

### 2. Job Matching
1. Parse resumes first (step 1)
2. Go to "Job Matching" tab
3. Enter job title and paste complete job description
4. Adjust scoring weights as needed:
   - **Skills Weight**: Importance of technical skills match
   - **Experience Weight**: Importance of years of experience
   - **Education Weight**: Importance of educational background
   - **Overall Fit Weight**: Semantic similarity and cultural fit
5. Click "🔍 Match & Rank"
6. View ranked results with detailed scores

### 3. Analytics
1. Complete job matching (steps 1-2)
2. Visit "Analytics" tab to see:
   - Score distribution charts
   - Skills vs experience correlation
   - Word clouds of candidate skills
   - Summary statistics

## How It Works

### Dynamic Resume Parsing
The application uses advanced NLP techniques to extract information without predefined templates:

1. **Text Extraction**: Multi-library approach (pdfplumber, PyPDF2) for robust PDF parsing
2. **Entity Recognition**: spaCy's NER models identify names, organizations, and other entities
3. **Pattern Matching**: Intelligent regex patterns for emails, phones, and experience
4. **Semantic Analysis**: Contextual understanding of skills and qualifications
5. **Skills Database**: Comprehensive technical skills matching against 200+ technologies

### Intelligent Job Matching
The job matching system works like ChatGPT's analysis:

1. **Requirement Extraction**: NLP analysis of job descriptions to identify must-have vs nice-to-have skills
2. **Dynamic Benchmarking**: Automatic generation of scoring criteria based on job text
3. **Multi-dimensional Scoring**:
   - Skills matching (required vs preferred)
   - Experience level analysis
   - Education requirements verification
   - Semantic similarity for cultural fit
4. **Weighted Ranking**: Customizable importance weights for different criteria
5. **Detailed Reporting**: Gap analysis and improvement recommendations

## Sample Output Structure

### Parsed Resume JSON
```json
{
  "name": "John Doe",
  "email": "john.doe@email.com",
  "phone": "+1-555-0123",
  "skills": ["Python", "React", "AWS", "Docker", "PostgreSQL"],
  "education": ["Bachelor of Computer Science", "University of Technology"],
  "experience_years": 5,
  "companies": ["Tech Corp Inc", "StartupXYZ"],
  "certifications": ["AWS Certified Developer"],
  "projects": ["E-commerce Platform", "ML Recommendation System"],
  "summary": "Experienced software developer with 5 years...",
  "parsed_at": "2024-01-15T10:30:00"
}
```

### Job Match Results
```json
{
  "filename": "john_doe_resume.pdf",
  "overall_score": 0.823,
  "scores": {
    "skills": 0.875,
    "experience": 0.900,
    "education": 0.750,
    "overall_fit": 0.765
  },
  "match_details": {
    "matched_skills": ["Python", "React", "AWS"],
    "missing_skills": ["Kubernetes", "GraphQL"],
    "experience_gap": 0,
    "recommendations": ["Consider gaining experience in Kubernetes"]
  }
}
```

## Configuration

### Adjustable Parameters
- **Score Thresholds**: Modify `SCORE_THRESHOLDS` in `utils.py`
- **Skill Database**: Extend `skill_categories` in `job_matcher.py`
- **File Limits**: Adjust `MAX_FILE_SIZE` and `MAX_FILES` in `utils.py`
- **Model Settings**: Change transformer models in `MODEL_CONFIGS`

### Custom Skills Database
Add industry-specific skills by modifying the `skill_categories` dictionary:

```python
'blockchain': ['solidity', 'ethereum', 'smart contracts', 'defi', 'web3'],
'mobile': ['react native', 'flutter', 'swift', 'kotlin', 'ionic']
```

## Performance Notes

- **First Run**: Initial model loading takes 30-60 seconds
- **Processing Speed**: ~2-3 seconds per resume for parsing
- **Memory Usage**: ~1-2GB RAM for model storage
- **Batch Size**: Recommended maximum 20 resumes per batch

## Troubleshooting

### Common Issues

1. **Models Not Loading**:
   ```bash
   python install_models.py
   ```

2. **PDF Parsing Errors**:
   - Try different PDF files
   - Check if PDF contains extractable text (not scanned images)

3. **Memory Issues**:
   - Process fewer files at once
   - Restart the application if models become unresponsive

4. **Port Already in Use**:
   ```bash
   streamlit run app.py --server.port 8502
   ```

### Performance Optimization
- Use SSD storage for better model loading
- Close other applications to free RAM
- Process resumes in smaller batches for large datasets

## Contributing

To extend the application:

1. **Add New File Formats**: Extend `extract_text()` in `resume_parser.py`
2. **Improve Skills Detection**: Update skill databases in both parser and matcher
3. **Custom Scoring**: Modify scoring algorithms in `job_matcher.py`
4. **UI Enhancements**: Update Streamlit components in `app.py`

## License

This project is for educational and personal use. Please ensure compliance with data privacy regulations when processing resumes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed correctly
3. Ensure Python version compatibility (3.8+)

---

**Note**: This application processes sensitive personal information. Always ensure proper data handling and privacy compliance when using with real resumes.