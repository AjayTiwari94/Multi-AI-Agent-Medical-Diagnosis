# Configuration settings for Medical Diagnostics AI Agent

import os
from dotenv import load_dotenv

# Load environment variables (ignore if file doesn't exist)
try:
    load_dotenv()
except (FileNotFoundError, UnicodeDecodeError):
    print("Warning: .env file not found or corrupted. Using default configuration.")

class Config:
    # API Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your_gemini_api_key_here')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # App Configuration
    APP_TITLE = "AI Medical Diagnostics Assistant"
    APP_ICON = "🩺"
    
    # Model Configuration
    GEMINI_MODEL = "gemini-2.0-flash"  # Switched to Flash for higher quotas
    GEMINI_MODEL_FAST = "gemini-2.0-flash"  # Faster model with lower quota usage
    GEMINI_MODEL_PRO = "gemini-2.0-pro"  # Best quality but higher quota usage
    
    # UI Configuration
    SIDEBAR_STATE = "expanded"
    LAYOUT = "wide"
    
    # Medical Disclaimer
    MEDICAL_DISCLAIMER = """
    ⚠️ **Medical Disclaimer**
    
    This AI assistant is for informational purposes only and does not replace 
    professional medical advice, diagnosis, or treatment.
    
    Always consult with qualified healthcare professionals for medical decisions.
    """
    
    # System Prompts
    DIAGNOSTIC_SYSTEM_PROMPT = """
    You are an AI Medical Diagnostic Assistant. Your role is to:
    
    1. Analyze medical reports and symptoms provided by users
    2. Provide informative medical insights based on the data
    3. Suggest possible conditions or areas of concern
    4. Recommend when to seek professional medical advice
    
    IMPORTANT DISCLAIMERS:
    - You are NOT a replacement for professional medical diagnosis
    - Always recommend consulting with healthcare professionals
    - Your insights are for informational purposes only
    - Do not provide specific treatment recommendations
    
    When analyzing reports:
    - Be thorough and systematic
    - Explain medical terms in simple language
    - Highlight any concerning findings
    - Provide educational context about conditions
    - Always emphasize the importance of professional medical consultation
    """
