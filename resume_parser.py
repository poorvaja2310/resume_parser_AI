import re
import PyPDF2
import pdfplumber
import docx
import nltk
import spacy
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

class ResumeParser:
    def __init__(self, models):
        self.nlp = models['spacy']
        self.sentence_model = models['sentence_transformer']
        self.setup_patterns()
    
    def setup_patterns(self):
        """Setup regex patterns for information extraction"""
        # Email pattern
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # Phone pattern
        self.phone_pattern = re.compile(r'[\+]?[1-9]?[\d\s\-\(\)]{7,15}')
        
        # Years of experience patterns
        self.experience_patterns = [
            re.compile(r'(\d+)[\+\s]*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)', re.IGNORECASE),
            re.compile(r'(?:experience|exp)[\s\:\-]*(\d+)[\+\s]*(?:years?|yrs?)', re.IGNORECASE),
            re.compile(r'(\d+)[\+\s]*(?:years?|yrs?)', re.IGNORECASE)
        ]
        
        # Education keywords
        self.education_keywords = [
            'bachelor', 'master', 'phd', 'doctorate', 'degree', 'university', 'college',
            'btech', 'mtech', 'bsc', 'msc', 'bca', 'mca', 'bba', 'mba', 'be', 'me',
            'engineering', 'computer science', 'information technology', 'diploma'
        ]
        
        # Skills database (expanded)
        self.technical_skills = [
            # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
            'scala', 'kotlin', 'swift', 'objective-c', 'r', 'matlab', 'perl', 'shell', 'bash',
            
            # Web Technologies
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask',
            'spring', 'asp.net', 'laravel', 'symfony', 'rails', 'jquery', 'bootstrap', 'sass',
            
            # Databases
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle', 'sql server',
            'sqlite', 'cassandra', 'dynamodb', 'neo4j', 'influxdb',
            
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible',
            'gitlab', 'github', 'circleci', 'travis ci', 'heroku', 'vercel', 'netlify',
            
            # Data Science & AI
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras', 'scikit-learn',
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'jupyter', 'spark', 'hadoop', 'tableau',
            'power bi', 'looker', 'airflow', 'mlflow', 'kubeflow',
            
            # Tools & Others
            'git', 'jira', 'confluence', 'slack', 'trello', 'postman', 'swagger', 'rest api',
            'graphql', 'microservices', 'agile', 'scrum', 'kanban', 'tdd', 'bdd', 'ci/cd'
        ]
        
        # Company indicators
        self.company_indicators = [
            'inc', 'corp', 'corporation', 'ltd', 'limited', 'llc', 'company', 'co',
            'technologies', 'tech', 'systems', 'solutions', 'services', 'consulting',
            'software', 'digital', 'labs', 'studios', 'group', 'enterprises'
        ]
    
    def parse_resume(self, uploaded_file):
        """Main parsing function"""
        try:
            # Extract text based on file type
            text = self.extract_text(uploaded_file)
            
            if not text or len(text.strip()) < 50:
                raise ValueError("Could not extract sufficient text from the resume")
            
            # Parse different sections
            parsed_data = {
                'raw_text': text,
                'name': self.extract_name(text),
                'email': self.extract_email(text),
                'phone': self.extract_phone(text),
                'skills': self.extract_skills(text),
                'education': self.extract_education(text),
                'experience_years': self.extract_experience_years(text),
                'companies': self.extract_companies(text),
                'certifications': self.extract_certifications(text),
                'projects': self.extract_projects(text),
                'summary': self.generate_summary(text),
                'parsed_at': datetime.now().isoformat()
            }
            
            return parsed_data
            
        except Exception as e:
            raise Exception(f"Error parsing resume: {str(e)}")
    
    def extract_text(self, uploaded_file):
        """Extract text from different file formats"""
        try:
            if uploaded_file.type == "application/pdf":
                return self.extract_pdf_text(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                return self.extract_docx_text(uploaded_file)
            elif uploaded_file.type == "text/plain":
                return str(uploaded_file.read(), "utf-8")
            else:
                raise ValueError(f"Unsupported file type: {uploaded_file.type}")
        except Exception as e:
            raise Exception(f"Text extraction failed: {str(e)}")
    
    def extract_pdf_text(self, uploaded_file):
        """Extract text from PDF using multiple methods"""
        text = ""
        
        # Try pdfplumber first (better for complex layouts)
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except:
            pass
        
        # Fallback to PyPDF2 if pdfplumber fails
        if not text or len(text.strip()) < 100:
            try:
                uploaded_file.seek(0)  # Reset file pointer
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            except:
                pass
        
        return text.strip()
    
    def extract_docx_text(self, uploaded_file):
        """Extract text from DOCX"""
        try:
            doc = docx.Document(uploaded_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"DOCX extraction failed: {str(e)}")
    
    def extract_name(self, text):
        """Extract candidate name using NLP"""
        try:
            # Use spaCy to find person entities
            doc = self.nlp(text[:500])  # Check first 500 characters
            
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    # Validate that it's likely a real name
                    name_words = ent.text.split()
                    if len(name_words) >= 2 and len(ent.text) < 50:
                        return ent.text.strip()
            
            # Fallback: Look for name patterns at the beginning
            lines = text.split('\n')[:5]  # Check first 5 lines
            for line in lines:
                line = line.strip()
                # Look for lines that might be names (2-4 words, proper case)
                words = line.split()
                if 2 <= len(words) <= 4 and all(word[0].isupper() for word in words if word):
                    return line
            
            return "Not found"
        except:
            return "Not found"
    
    def extract_email(self, text):
        """Extract email address"""
        try:
            matches = self.email_pattern.findall(text.lower())
            return matches[0] if matches else "Not found"
        except:
            return "Not found"
    
    def extract_phone(self, text):
        """Extract phone number"""
        try:
            matches = self.phone_pattern.findall(text)
            if matches:
                # Clean and return the first valid-looking phone number
                for match in matches:
                    cleaned = re.sub(r'[^\d+]', '', match)
                    if len(cleaned) >= 7:  # Minimum valid phone length
                        return match.strip()
            return "Not found"
        except:
            return "Not found"
    
    def extract_skills(self, text):
        """Extract technical skills"""
        try:
            text_lower = text.lower()
            found_skills = []
            
            # Check for each skill in our database
            for skill in self.technical_skills:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    found_skills.append(skill)
            
            # Also use NLP to find technology-related entities
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT"] and ent.text.lower() in text_lower:
                    skill_candidate = ent.text.lower()
                    if skill_candidate not in [s.lower() for s in found_skills]:
                        if any(tech_word in skill_candidate for tech_word in 
                              ['tech', 'soft', 'program', 'develop', 'engineer', 'framework', 'library']):
                            found_skills.append(ent.text)
            
            return list(set(found_skills))  # Remove duplicates
        except:
            return []
    
    def extract_education(self, text):
        """Extract education information"""
        try:
            text_lower = text.lower()
            education_info = []
            
            # Split text into sentences/lines
            sentences = re.split(r'[.\n]', text)
            
            for sentence in sentences:
                sentence_lower = sentence.lower().strip()
                
                # Check if sentence contains education keywords
                if any(keyword in sentence_lower for keyword in self.education_keywords):
                    if len(sentence.strip()) > 10 and len(sentence.strip()) < 200:
                        education_info.append(sentence.strip())
            
            # Remove duplicates and limit results
            education_info = list(set(education_info))[:5]
            
            return education_info if education_info else ["Not found"]
        except:
            return ["Not found"]
    
    def extract_experience_years(self, text):
        """Extract years of experience"""
        try:
            years = []
            
            for pattern in self.experience_patterns:
                matches = pattern.findall(text)
                years.extend([int(match) for match in matches if match.isdigit()])
            
            if years:
                # Return the maximum years found (most comprehensive experience)
                return max(years)
            
            # Alternative: Count job positions and estimate
            job_indicators = ['developer', 'engineer', 'analyst', 'manager', 'specialist', 
                            'consultant', 'architect', 'lead', 'senior', 'junior']
            job_count = sum(1 for indicator in job_indicators if indicator in text.lower())
            
            if job_count > 0:
                return min(job_count * 2, 15)  # Estimate 2 years per position, max 15
            
            return "Not calculated"
        except:
            return "Not calculated"
    
    def extract_companies(self, text):
        """Extract company names"""
        try:
            companies = []
            
            # Use NLP to find organization entities
            doc = self.nlp(text)
            
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    company_name = ent.text.strip()
                    # Filter out common false positives
                    if (len(company_name) > 2 and 
                        company_name.lower() not in ['resume', 'cv', 'experience', 'education', 'skills'] and
                        not company_name.isdigit()):
                        companies.append(company_name)
            
            # Also look for patterns with company indicators
            sentences = re.split(r'[.\n]', text)
            for sentence in sentences:
                words = sentence.split()
                for i, word in enumerate(words):
                    if word.lower() in self.company_indicators:
                        # Check if there's a potential company name before this indicator
                        if i > 0:
                            potential_company = ' '.join(words[max(0, i-3):i+1])
                            if len(potential_company) < 100:
                                companies.append(potential_company.strip())
            
            # Remove duplicates and limit results
            companies = list(set(companies))[:8]
            
            return companies if companies else ["Not found"]
        except:
            return ["Not found"]
    
    def extract_certifications(self, text):
        """Extract certifications"""
        try:
            cert_keywords = [
                'certified', 'certification', 'certificate', 'aws', 'azure', 'google cloud',
                'cisco', 'microsoft', 'oracle', 'scrum master', 'pmp', 'comptia', 'cissp'
            ]
            
            certifications = []
            sentences = re.split(r'[.\n]', text)
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in cert_keywords):
                    if 10 < len(sentence.strip()) < 150:
                        certifications.append(sentence.strip())
            
            return list(set(certifications))[:5] if certifications else ["Not found"]
        except:
            return ["Not found"]
    
    def extract_projects(self, text):
        """Extract project information"""
        try:
            project_keywords = ['project', 'developed', 'built', 'created', 'implemented', 'designed']
            
            projects = []
            sentences = re.split(r'[.\n]', text)
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in project_keywords):
                    if 20 < len(sentence.strip()) < 300:
                        projects.append(sentence.strip())
            
            return list(set(projects))[:5] if projects else ["Not found"]
        except:
            return ["Not found"]
    
    def generate_summary(self, text):
        """Generate a brief summary of the resume"""
        try:
            # Take key sentences and create summary
            sentences = nltk.sent_tokenize(text)[:10]  # First 10 sentences
            
            if len(sentences) > 3:
                summary = '. '.join(sentences[:3]) + '.'
                return summary[:500] + "..." if len(summary) > 500 else summary
            else:
                return text[:500] + "..." if len(text) > 500 else text
        except:
            return "Summary not generated"
