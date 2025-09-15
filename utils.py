# import nltk
# import spacy
# from sentence_transformers import SentenceTransformer
# import streamlit as st
# import os

# @st.cache_resource
# def setup_nltk_data():
#     """Download required NLTK data"""
#     try:
#         nltk.data.find('tokenizers/punkt')
#     except LookupError:
#         nltk.download('punkt')
    
#     try:
#         nltk.data.find('corpora/stopwords')
#     except LookupError:
#         nltk.download('stopwords')
    
#     try:
#         nltk.data.find('tokenizers/punkt_tab')
#     except LookupError:
#         nltk.download('punkt_tab')
    
#     return True

# @st.cache_resource
# def load_models():
#     """Load and cache all required models"""
#     models = {}
    
#     # Load spaCy model
#     try:
#         models['spacy'] = spacy.load("en_core_web_sm")
#     except OSError:
#         st.error("""
#         spaCy English model not found. Please install it using:
#         ```
#         python -m spacy download en_core_web_sm
#         ```
#         """)
#         st.stop()
    
#     # Load sentence transformer
#     try:
#         models['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
#     except Exception as e:
#         st.error(f"Failed to load sentence transformer: {e}")
#         st.stop()
    
#     return models

# def clean_text(text):
#     """Clean and preprocess text"""
#     if not text:
#         return ""
    
#     # Remove extra whitespace
#     text = ' '.join(text.split())
    
#     # Remove special characters but keep important punctuation
#     import re
#     text = re.sub(r'[^\w\s\.\,\-\(\)]', ' ', text)
    
#     return text.strip()

# def extract_sections(text):
#     """Extract different sections from resume text"""
#     sections = {}
    
#     # Common section headers
#     section_patterns = {
#         'summary': r'(?:summary|profile|objective|about)',
#         'experience': r'(?:experience|employment|work history|professional experience)',
#         'education': r'(?:education|academic|qualification)',
#         'skills': r'(?:skills|technical skills|competencies|expertise)',
#         'projects': r'(?:projects|portfolio)',
#         'certifications': r'(?:certification|certificate|license)'
#     }
    
#     lines = text.split('\n')
#     current_section = None
    
#     for line in lines:
#         line_lower = line.lower().strip()
        
#         # Check if line is a section header
#         for section_name, pattern in section_patterns.items():
#             import re
#             if re.search(pattern, line_lower) and len(line.strip()) < 100:
#                 current_section = section_name
#                 sections[section_name] = []
#                 break
#         else:
#             # Add line to current section
#             if current_section and line.strip():
#                 sections[current_section].append(line.strip())
    
#     # Convert lists to strings
#     for section, content in sections.items():
#         sections[section] = '\n'.join(content)
    
#     return sections

# def calculate_text_similarity(text1, text2, model=None):
#     """Calculate semantic similarity between two texts"""
#     try:
#         if model is None:
#             model = SentenceTransformer('all-MiniLM-L6-v2')
        
#         if not text1 or not text2:
#             return 0.0
        
#         embeddings = model.encode([text1, text2])
        
#         from sklearn.metrics.pairwise import cosine_similarity
#         import numpy as np
        
#         similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
#         return float(similarity)
        
#     except Exception:
#         return 0.0

# def format_score(score):
#     """Format score for display"""
#     if isinstance(score, (int, float)):
#         return f"{score:.2f}"
#     return str(score)

# def get_match_color(score):
#     """Get color based on match score"""
#     if score >= 0.7:
#         return "success"
#     elif score >= 0.5:
#         return "warning"
#     else:
#         return "error"

# def validate_file_type(file):
#     """Validate uploaded file type"""
#     allowed_types = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain']
#     return file.type in allowed_types

# def safe_extract(data, key, default="Not found"):
#     """Safely extract data with default fallback"""
#     try:
#         value = data.get(key, default)
#         return value if value else default
#     except:
#         return default

# def create_progress_bar(current, total, label="Processing"):
#     """Create a progress indicator"""
#     progress = current / total if total > 0 else 0
#     return f"{label}: {current}/{total} ({progress:.1%})"

# # Configuration constants
# MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
# MAX_FILES = 20
# SUPPORTED_FORMATS = ['pdf', 'docx', 'txt']

# # Model configurations
# MODEL_CONFIGS = {
#     'sentence_transformer': {
#         'model_name': 'all-MiniLM-L6-v2',
#         'cache_dir': './models/sentence_transformers'
#     },
#     'spacy': {
#         'model_name': 'en_core_web_sm'
#     }
# }

# # Scoring weights (can be adjusted)
# DEFAULT_WEIGHTS = {
#     'skills': 0.4,
#     'experience': 0.3,
#     'education': 0.2,
#     'overall_fit': 0.1
# }

# # Threshold scores
# SCORE_THRESHOLDS = {
#     'high_match': 0.7,
#     'medium_match': 0.5,
#     'low_match': 0.0
# }



import nltk
import spacy
from sentence_transformers import SentenceTransformer
import streamlit as st
import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def setup_nltk_data():
    """Download required NLTK data"""
    # Ensure punkt tokenizer
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Ensure stopwords
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    return True

@st.cache_resource
def load_models():
    """Load and cache all required models"""
    models = {}
    
    # Load spaCy model
    try:
        models['spacy'] = spacy.load("en_core_web_sm")
    except OSError:
        st.error("""
        spaCy English model not found. Please install it using:
        ```
        python -m spacy download en_core_web_sm
        ```
        """)
        st.stop()
    
    # Load sentence transformer
    try:
        models['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Failed to load sentence transformer: {e}")
        st.stop()
    
    return models

def clean_text(text):
    """Clean and preprocess text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s\.\,\-\(\)]', ' ', text)
    
    return text.strip()

def extract_sections(text):
    """Extract different sections from resume text"""
    sections = {}
    
    section_patterns = {
        'summary': r'(?:summary|profile|objective|about)',
        'experience': r'(?:experience|employment|work history|professional experience)',
        'education': r'(?:education|academic|qualification)',
        'skills': r'(?:skills|technical skills|competencies|expertise)',
        'projects': r'(?:projects|portfolio)',
        'certifications': r'(?:certification|certificate|license)'
    }
    
    lines = text.split('\n')
    current_section = None
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Check if line is a section header
        for section_name, pattern in section_patterns.items():
            if re.search(pattern, line_lower) and len(line.strip()) < 100:
                current_section = section_name
                sections[section_name] = []
                break
        else:
            # Add line to current section
            if current_section and line.strip():
                sections[current_section].append(line.strip())
    
    # Convert lists to strings
    for section, content in sections.items():
        sections[section] = '\n'.join(content)
    
    return sections

def calculate_text_similarity(text1, text2, model=None):
    """Calculate semantic similarity between two texts"""
    try:
        if model is None:
            model = SentenceTransformer('all-MiniLM-L6-v2')
        
        if not text1 or not text2:
            return 0.0
        
        embeddings = model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
        
    except Exception:
        return 0.0

def format_score(score):
    """Format score for display"""
    if isinstance(score, (int, float)):
        return f"{score:.2f}"
    return str(score)

def get_match_color(score):
    """Get color based on match score"""
    if score >= 0.7:
        return "success"
    elif score >= 0.5:
        return "warning"
    else:
        return "error"

def validate_file_type(file):
    """Validate uploaded file type"""
    allowed_types = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain']
    return file.type in allowed_types

def safe_extract(data, key, default="Not found"):
    """Safely extract data with default fallback"""
    try:
        value = data.get(key, default)
        return value if value else default
    except:
        return default

def create_progress_bar(current, total, label="Processing"):
    """Create a progress indicator"""
    progress = current / total if total > 0 else 0
    return f"{label}: {current}/{total} ({progress:.1%})"

# Configuration constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_FILES = 20
SUPPORTED_FORMATS = ['pdf', 'docx', 'txt']

# Model configurations
MODEL_CONFIGS = {
    'sentence_transformer': {
        'model_name': 'all-MiniLM-L6-v2',
        'cache_dir': './models/sentence_transformers'
    },
    'spacy': {
        'model_name': 'en_core_web_sm'
    }
}

# Scoring weights (can be adjusted)
DEFAULT_WEIGHTS = {
    'skills': 0.4,
    'experience': 0.3,
    'education': 0.2,
    'overall_fit': 0.1
}

# Threshold scores
SCORE_THRESHOLDS = {
    'high_match': 0.7,
    'medium_match': 0.5,
    'low_match': 0.0
}
