#!/usr/bin/env python3
"""
Script to install required models for the resume parser
Run this before starting the main application
"""

import subprocess
import sys
import os

def install_spacy_model():
    """Install spaCy English model"""
    try:
        print("Installing spaCy English model...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                      check=True, capture_output=True, text=True)
        print("✅ spaCy model installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install spaCy model: {e}")
        print("Please run manually: python -m spacy download en_core_web_sm")
        return False
    return True

def install_nltk_data():
    """Install required NLTK data"""
    try:
        import nltk
        print("Installing NLTK data...")
        
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        
        print("✅ NLTK data installed successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to install NLTK data: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Not compatible")
        print("Please use Python 3.8 or higher")
        return False

def install_requirements():
    """Install Python package requirements"""
    try:
        print("Installing Python packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Python packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def main():
    """Main installation function"""
    print("🚀 Setting up Resume Parser Application...")
    print("=" * 50)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Install requirements
    if success and not install_requirements():
        success = False
    
    # Install NLTK data
    if success and not install_nltk_data():
        success = False
    
    # Install spaCy model
    if success and not install_spacy_model():
        success = False
    
    print("=" * 50)
    if success:
        print("🎉 Setup completed successfully!")
        print("\nTo start the application, run:")
        print("streamlit run app.py")
    else:
        print("❌ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()