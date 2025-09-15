import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

class JobMatcher:
    def __init__(self, models):
        self.nlp = models['spacy']
        self.sentence_model = models['sentence_transformer']
        self.stop_words = set(stopwords.words('english'))
        
        # Technical skills database (same as in resume parser but organized by category)
        self.skill_categories = {
            'programming': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 
                'go', 'rust', 'scala', 'kotlin', 'swift', 'r', 'matlab'
            ],
            'web': [
                'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 
                'flask', 'spring', 'asp.net', 'laravel', 'jquery', 'bootstrap'
            ],
            'database': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle', 
                'sql server', 'sqlite', 'cassandra', 'dynamodb'
            ],
            'cloud': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 
                'ansible', 'heroku', 'vercel'
            ],
            'data_science': [
                'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras', 
                'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'spark', 'hadoop'
            ],
            'tools': [
                'git', 'jira', 'confluence', 'postman', 'swagger', 'rest api', 'graphql', 
                'microservices', 'agile', 'scrum'
            ]
        }
        
        # Experience level mapping
        self.experience_levels = {
            'junior': {'years': (0, 2), 'keywords': ['junior', 'entry', 'graduate', 'trainee', 'intern']},
            'mid': {'years': (2, 5), 'keywords': ['mid', 'intermediate', 'associate', 'developer']},
            'senior': {'years': (5, 10), 'keywords': ['senior', 'lead', 'principal', 'expert']},
            'expert': {'years': (10, 100), 'keywords': ['architect', 'director', 'head', 'chief', 'vp']}
        }
        
        # Education levels
        self.education_levels = {
            'high_school': ['high school', 'diploma', '12th'],
            'bachelor': ['bachelor', 'btech', 'bsc', 'bca', 'bba', 'be', 'undergraduate'],
            'master': ['master', 'mtech', 'msc', 'mca', 'mba', 'me', 'graduate'],
            'doctorate': ['phd', 'doctorate', 'doctoral', 'postgraduate']
        }
    
    def generate_job_benchmark(self, job_description, job_title=""):
        """
        Dynamically generate job requirements benchmark using AI-like analysis
        Similar to how ChatGPT would analyze a job description
        """
        try:
            # Clean and prepare text
            full_text = f"{job_title} {job_description}".lower()
            
            # Extract key requirements using NLP
            benchmark = {
                'job_title': job_title,
                'required_skills': self._extract_required_skills(full_text),
                'preferred_skills': self._extract_preferred_skills(full_text),
                'experience_requirements': self._extract_experience_requirements(full_text),
                'education_requirements': self._extract_education_requirements(full_text),
                'soft_skills': self._extract_soft_skills(full_text),
                'responsibilities': self._extract_responsibilities(full_text),
                'company_type': self._infer_company_type(full_text),
                'role_level': self._infer_role_level(full_text, job_title),
                'domain': self._infer_domain(full_text),
                'work_type': self._infer_work_type(full_text),
                'urgency_level': self._assess_urgency(full_text),
                'skill_weights': self._calculate_skill_weights(full_text),
                'generated_at': 'dynamic'
            }
            
            return benchmark
            
        except Exception as e:
            # Fallback benchmark
            return self._create_fallback_benchmark(job_title, job_description)
    
    def _extract_required_skills(self, text):
        """Extract must-have technical skills"""
        required_skills = []
        
        # Look for explicit requirements
        requirement_patterns = [
            r'required?\s*:?\s*([^.]*)',
            r'must\s+have\s*:?\s*([^.]*)',
            r'essential\s*:?\s*([^.]*)',
            r'mandatory\s*:?\s*([^.]*)',
            r'minimum\s*:?\s*([^.]*)'
        ]
        
        for pattern in requirement_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                skills = self._extract_skills_from_text(match)
                required_skills.extend(skills)
        
        # Check against our skill database
        all_skills = []
        for category, skills in self.skill_categories.items():
            all_skills.extend(skills)
        
        found_skills = []
        for skill in all_skills:
            if skill.lower() in text:
                # Check context to determine if it's required
                skill_context = self._get_skill_context(text, skill)
                if any(req_word in skill_context for req_word in 
                      ['required', 'must', 'essential', 'mandatory', 'minimum']):
                    found_skills.append(skill)
        
        # Remove duplicates and return top skills
        required_skills.extend(found_skills)
        return list(set(required_skills))[:10]
    
    def _extract_preferred_skills(self, text):
        """Extract nice-to-have skills"""
        preferred_skills = []
        
        preference_patterns = [
            r'preferred?\s*:?\s*([^.]*)',
            r'nice\s+to\s+have\s*:?\s*([^.]*)',
            r'bonus\s*:?\s*([^.]*)',
            r'additional\s*:?\s*([^.]*)',
            r'plus\s*:?\s*([^.]*)'
        ]
        
        for pattern in preference_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                skills = self._extract_skills_from_text(match)
                preferred_skills.extend(skills)
        
        return list(set(preferred_skills))[:8]
    
    def _extract_experience_requirements(self, text):
        """Extract experience requirements"""
        experience_patterns = [
            r'(\d+)[\+\s]*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'(?:experience|exp)[\s\:\-]*(\d+)[\+\s]*(?:years?|yrs?)',
            r'minimum\s+(\d+)\s+years?',
            r'at\s+least\s+(\d+)\s+years?'
        ]
        
        years_found = []
        for pattern in experience_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            years_found.extend([int(match) for match in matches])
        
        if years_found:
            min_years = min(years_found)
            max_years = max(years_found) if len(years_found) > 1 else min_years + 2
            return {
                'min_years': min_years,
                'max_years': max_years,
                'level': self._years_to_level(min_years)
            }
        
        # Infer from role title
        level = self._infer_role_level(text, "")
        if level == 'junior':
            return {'min_years': 0, 'max_years': 2, 'level': 'junior'}
        elif level == 'senior':
            return {'min_years': 5, 'max_years': 10, 'level': 'senior'}
        else:
            return {'min_years': 2, 'max_years': 5, 'level': 'mid'}
    
    def _extract_education_requirements(self, text):
        """Extract education requirements"""
        education_req = []
        
        for level, keywords in self.education_levels.items():
            for keyword in keywords:
                if keyword in text:
                    education_req.append(level)
        
        # Determine minimum required level
        if 'doctorate' in education_req:
            return {'min_level': 'doctorate', 'preferred': 'doctorate'}
        elif 'master' in education_req:
            return {'min_level': 'master', 'preferred': 'master'}
        elif 'bachelor' in education_req:
            return {'min_level': 'bachelor', 'preferred': 'bachelor'}
        else:
            return {'min_level': 'high_school', 'preferred': 'bachelor'}
    
    def _extract_soft_skills(self, text):
        """Extract soft skills and personality traits"""
        soft_skills_db = [
            'leadership', 'communication', 'teamwork', 'problem solving', 'analytical',
            'creative', 'innovative', 'adaptable', 'flexible', 'collaborative',
            'detail oriented', 'organized', 'self motivated', 'proactive', 'initiative'
        ]
        
        found_soft_skills = []
        for skill in soft_skills_db:
            if skill.lower() in text:
                found_soft_skills.append(skill)
        
        return found_soft_skills[:8]
    
    def _extract_responsibilities(self, text):
        """Extract key responsibilities"""
        responsibility_patterns = [
            r'responsibilities?\s*:?\s*([^.]*)',
            r'duties?\s*:?\s*([^.]*)',
            r'you\s+will\s+([^.]*)',
            r'candidate\s+will\s+([^.]*)'
        ]
        
        responsibilities = []
        for pattern in responsibility_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            responsibilities.extend([match.strip() for match in matches if len(match.strip()) > 10])
        
        return responsibilities[:5]
    
    def _infer_company_type(self, text):
        """Infer company type from job description"""
        if any(word in text for word in ['startup', 'fast-paced', 'dynamic', 'agile']):
            return 'startup'
        elif any(word in text for word in ['enterprise', 'fortune', 'established', 'corporate']):
            return 'enterprise'
        elif any(word in text for word in ['consulting', 'client', 'project-based']):
            return 'consulting'
        else:
            return 'unknown'
    
    def _infer_role_level(self, text, job_title):
        """Infer role level from title and description"""
        combined_text = f"{job_title} {text}".lower()
        
        for level, info in self.experience_levels.items():
            if any(keyword in combined_text for keyword in info['keywords']):
                return level
        
        return 'mid'  # Default
    
    def _infer_domain(self, text):
        """Infer industry domain"""
        domains = {
            'fintech': ['finance', 'banking', 'payment', 'trading', 'investment'],
            'healthcare': ['healthcare', 'medical', 'hospital', 'patient', 'clinical'],
            'ecommerce': ['ecommerce', 'retail', 'shopping', 'marketplace', 'commerce'],
            'edtech': ['education', 'learning', 'student', 'course', 'academic'],
            'saas': ['saas', 'software as a service', 'cloud', 'subscription'],
            'gaming': ['game', 'gaming', 'entertainment', 'mobile games'],
            'iot': ['iot', 'internet of things', 'sensors', 'embedded'],
            'ai': ['artificial intelligence', 'machine learning', 'ai', 'ml', 'nlp']
        }
        
        for domain, keywords in domains.items():
            if any(keyword in text for keyword in keywords):
                return domain
        
        return 'technology'
    
    def _infer_work_type(self, text):
        """Infer work arrangement"""
        if 'remote' in text:
            return 'remote'
        elif 'hybrid' in text:
            return 'hybrid'
        elif 'on-site' in text or 'office' in text:
            return 'onsite'
        else:
            return 'unspecified'
    
    def _assess_urgency(self, text):
        """Assess hiring urgency"""
        urgency_indicators = ['urgent', 'immediate', 'asap', 'quickly', 'fast']
        urgency_score = sum(1 for indicator in urgency_indicators if indicator in text)
        
        if urgency_score >= 2:
            return 'high'
        elif urgency_score == 1:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_skill_weights(self, text):
        """Calculate weights for different skill categories based on job description"""
        category_mentions = {}
        total_mentions = 0
        
        for category, skills in self.skill_categories.items():
            mentions = sum(1 for skill in skills if skill.lower() in text)
            category_mentions[category] = mentions
            total_mentions += mentions
        
        if total_mentions == 0:
            # Default weights
            return {
                'programming': 0.3,
                'web': 0.2,
                'database': 0.15,
                'cloud': 0.15,
                'data_science': 0.1,
                'tools': 0.1
            }
        
        # Normalize weights
        weights = {}
        for category, mentions in category_mentions.items():
            weights[category] = mentions / total_mentions
        
        return weights
    
    def _extract_skills_from_text(self, text):
        """Extract skills from a piece of text"""
        skills = []
        text_lower = text.lower()
        
        for category, skill_list in self.skill_categories.items():
            for skill in skill_list:
                if skill.lower() in text_lower:
                    skills.append(skill)
        
        return skills
    
    def _get_skill_context(self, text, skill):
        """Get context around a skill mention"""
        skill_pos = text.lower().find(skill.lower())
        if skill_pos == -1:
            return ""
        
        start = max(0, skill_pos - 50)
        end = min(len(text), skill_pos + len(skill) + 50)
        
        return text[start:end].lower()
    
    def _years_to_level(self, years):
        """Convert years to experience level"""
        if years <= 2:
            return 'junior'
        elif years <= 5:
            return 'mid'
        elif years <= 10:
            return 'senior'
        else:
            return 'expert'
    
    def _create_fallback_benchmark(self, job_title, job_description):
        """Create a basic benchmark if parsing fails"""
        return {
            'job_title': job_title,
            'required_skills': ['programming', 'problem solving'],
            'preferred_skills': ['communication', 'teamwork'],
            'experience_requirements': {'min_years': 2, 'max_years': 5, 'level': 'mid'},
            'education_requirements': {'min_level': 'bachelor', 'preferred': 'bachelor'},
            'soft_skills': ['communication', 'teamwork'],
            'responsibilities': ['develop software', 'collaborate with team'],
            'company_type': 'unknown',
            'role_level': 'mid',
            'domain': 'technology',
            'work_type': 'unspecified',
            'urgency_level': 'medium',
            'skill_weights': {'programming': 0.4, 'communication': 0.3, 'experience': 0.3},
            'generated_at': 'fallback'
        }
    
    def rank_resumes(self, resumes, job_benchmark, weights):
        """Rank resumes against job requirements"""
        try:
            ranked_resumes = []
            
            for resume in resumes:
                resume_data = resume['data']
                
                # Calculate individual scores
                skills_score = self._calculate_skills_match(resume_data, job_benchmark)
                experience_score = self._calculate_experience_match(resume_data, job_benchmark)
                education_score = self._calculate_education_match(resume_data, job_benchmark)
                overall_fit_score = self._calculate_overall_fit(resume_data, job_benchmark)
                
                # Calculate weighted overall score
                overall_score = (
                    skills_score * weights['skills'] +
                    experience_score * weights['experience'] +
                    education_score * weights['education'] +
                    overall_fit_score * weights['overall']
                )
                
                ranked_resume = {
                    'filename': resume['filename'],
                    'resume_data': resume_data,
                    'overall_score': round(overall_score, 3),
                    'scores': {
                        'skills': round(skills_score, 3),
                        'experience': round(experience_score, 3),
                        'education': round(education_score, 3),
                        'overall_fit': round(overall_fit_score, 3)
                    },
                    'match_details': self._generate_match_details(resume_data, job_benchmark)
                }
                
                ranked_resumes.append(ranked_resume)
            
            # Sort by overall score (descending)
            ranked_resumes.sort(key=lambda x: x['overall_score'], reverse=True)
            
            return ranked_resumes
            
        except Exception as e:
            raise Exception(f"Error ranking resumes: {str(e)}")
    
    def _calculate_skills_match(self, resume_data, job_benchmark):
        """Calculate skills matching score"""
        try:
            resume_skills = [skill.lower() for skill in resume_data.get('skills', [])]
            required_skills = [skill.lower() for skill in job_benchmark.get('required_skills', [])]
            preferred_skills = [skill.lower() for skill in job_benchmark.get('preferred_skills', [])]
            
            if not required_skills and not preferred_skills:
                return 0.5  # Neutral score if no requirements
            
            # Calculate matches
            required_matches = sum(1 for skill in required_skills if skill in resume_skills)
            preferred_matches = sum(1 for skill in preferred_skills if skill in resume_skills)
            
            # Score calculation
            required_score = required_matches / len(required_skills) if required_skills else 0
            preferred_score = preferred_matches / len(preferred_skills) if preferred_skills else 0
            
            # Weight required skills higher than preferred
            final_score = (required_score * 0.7) + (preferred_score * 0.3)
            
            return min(final_score, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_experience_match(self, resume_data, job_benchmark):
        """Calculate experience matching score"""
        try:
            resume_years = resume_data.get('experience_years')
            if resume_years == "Not calculated" or not isinstance(resume_years, (int, float)):
                return 0.3  # Neutral score if experience can't be determined
            
            experience_req = job_benchmark.get('experience_requirements', {})
            min_years = experience_req.get('min_years', 0)
            max_years = experience_req.get('max_years', 10)
            
            resume_years = float(resume_years)
            
            # Score based on how well experience fits the range
            if min_years <= resume_years <= max_years:
                return 1.0  # Perfect match
            elif resume_years > max_years:
                # Overqualified - still good but slightly lower score
                excess = resume_years - max_years
                return max(0.8 - (excess * 0.05), 0.6)
            else:
                # Underqualified
                shortfall = min_years - resume_years
                return max(0.3 - (shortfall * 0.1), 0.0)
                
        except Exception:
            return 0.0
    
    def _calculate_education_match(self, resume_data, job_benchmark):
        """Calculate education matching score"""
        try:
            resume_education = [edu.lower() for edu in resume_data.get('education', [])]
            education_req = job_benchmark.get('education_requirements', {})
            min_level = education_req.get('min_level', 'high_school')
            
            # Map education levels to scores
            level_scores = {
                'high_school': 1,
                'bachelor': 2,
                'master': 3,
                'doctorate': 4
            }
            
            min_score = level_scores.get(min_level, 1)
            
            # Check resume education level
            resume_score = 0
            for education in resume_education:
                for level, keywords in self.education_levels.items():
                    if any(keyword in education for keyword in keywords):
                        resume_score = max(resume_score, level_scores.get(level, 0))
            
            if resume_score == 0:
                resume_score = 1  # Assume high school if no education found
            
            # Calculate match score
            if resume_score >= min_score:
                return 1.0
            else:
                return resume_score / min_score
                
        except Exception:
            return 0.5
    
    def _calculate_overall_fit(self, resume_data, job_benchmark):
        """Calculate overall cultural and role fit using semantic analysis"""
        try:
            # Combine resume text for analysis
            resume_text = ""
            if resume_data.get('summary'):
                resume_text += resume_data['summary'] + " "
            if resume_data.get('raw_text'):
                resume_text += resume_data['raw_text'][:1000]  # First 1000 chars
            
            # Job requirements text
            job_text = ""
            if job_benchmark.get('responsibilities'):
                job_text += " ".join(job_benchmark['responsibilities']) + " "
            if job_benchmark.get('soft_skills'):
                job_text += " ".join(job_benchmark['soft_skills'])
            
            if not resume_text or not job_text:
                return 0.5
            
            # Use sentence transformer for semantic similarity
            resume_embedding = self.sentence_model.encode([resume_text])
            job_embedding = self.sentence_model.encode([job_text])
            
            similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
            
            return min(max(similarity, 0.0), 1.0)
            
        except Exception:
            return 0.5
    
    def _generate_match_details(self, resume_data, job_benchmark):
        """Generate detailed match analysis"""
        try:
            details = {
                'matched_skills': [],
                'missing_skills': [],
                'experience_gap': 0,
                'education_fit': 'unknown',
                'strengths': [],
                'recommendations': []
            }
            
            # Skills analysis
            resume_skills = [skill.lower() for skill in resume_data.get('skills', [])]
            required_skills = [skill.lower() for skill in job_benchmark.get('required_skills', [])]
            
            details['matched_skills'] = [skill for skill in required_skills if skill in resume_skills]
            details['missing_skills'] = [skill for skill in required_skills if skill not in resume_skills]
            
            # Experience analysis
            resume_years = resume_data.get('experience_years')
            if isinstance(resume_years, (int, float)):
                min_years = job_benchmark.get('experience_requirements', {}).get('min_years', 0)
                details['experience_gap'] = max(0, min_years - resume_years)
            
            # Generate recommendations
            if details['missing_skills']:
                details['recommendations'].append(f"Consider gaining experience in: {', '.join(details['missing_skills'][:3])}")
            
            if details['experience_gap'] > 0:
                details['recommendations'].append(f"Gain {details['experience_gap']} more years of relevant experience")
            
            return details
            
        except Exception:
            return {'error': 'Could not generate match details'}