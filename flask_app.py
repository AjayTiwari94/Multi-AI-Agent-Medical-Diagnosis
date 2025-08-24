#!/usr/bin/env python3
"""
Flask Web Application for Multi-Agent AI Medical Diagnostics System
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import google.generativeai as genai
import os
from dotenv import load_dotenv
import pandas as pd
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import PyPDF2
import io
from config import Config
from agents import MedicalAgentRouter, make_api_call_with_retry

# Load environment variables (ignore if file doesn't exist)
try:
    load_dotenv()
except (FileNotFoundError, UnicodeDecodeError):
    print("Warning: .env file not found or corrupted in flask_app.py. Using default configuration.")

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-this-in-production')

# Disable Flask's automatic .env loading
app.config['ENV'] = 'development'

# File upload configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# Global variables
agent_router = None
gemini_model = None

def configure_gemini():
    """Configure Google Gemini AI"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_gemini_api_key_here':
        return None
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(Config.GEMINI_MODEL)

def initialize_session():
    """Initialize session variables"""
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'diagnosis_history' not in session:
        session['diagnosis_history'] = []

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_stream):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

def extract_text_from_txt(file_stream):
    """Extract text from TXT file"""
    try:
        text = file_stream.read().decode('utf-8')
        return text.strip()
    except Exception as e:
        raise Exception(f"Error reading TXT file: {str(e)}")

def process_uploaded_file(file):
    """Process uploaded file and extract text"""
    if not file or not allowed_file(file.filename):
        raise Exception("Invalid file type. Please upload PDF or TXT files only.")
    
    filename = secure_filename(file.filename)
    file_extension = filename.rsplit('.', 1)[1].lower()
    
    # Reset file pointer to beginning
    file.seek(0)
    
    if file_extension == 'pdf':
        return extract_text_from_pdf(io.BytesIO(file.read()))
    elif file_extension == 'txt':
        return extract_text_from_txt(io.BytesIO(file.read()))
    else:
        raise Exception("Unsupported file type")

@app.before_request
def before_request():
    """Initialize session before each request"""
    initialize_session()

@app.route('/')
def home():
    """Home page showing all AI specialists"""
    global gemini_model, agent_router
    
    # Initialize Gemini model if not already done
    if gemini_model is None:
        gemini_model = configure_gemini()
    
    if agent_router is None and gemini_model:
        agent_router = MedicalAgentRouter(gemini_model)
    
    # Get system status
    system_status = {
        'gemini_configured': gemini_model is not None,
        'agents_ready': agent_router is not None,
        'total_analyses': len(session['diagnosis_history']),
        'chat_sessions': len(session['chat_history'])
    }
    
    # Get agent information
    agents_info = []
    if agent_router:
        for key, agent in agent_router.agents.items():
            agents_info.append({
                'key': key,
                'name': agent.specialty,
                'icon': agent.specialty_icon,
                'description': get_agent_description(key)
            })
    
    return render_template('home.html', 
                         system_status=system_status, 
                         agents_info=agents_info)

def get_agent_description(agent_key):
    """Get description for each agent"""
    descriptions = {
        'cardiology': 'Heart conditions, blood pressure, cardiovascular health, ECG analysis',
        'psychology': 'Mental health, depression, anxiety, stress management, sleep disorders',
        'pulmonology': 'Lung conditions, breathing issues, respiratory health, chest imaging',
        'general': 'Overall health, lab results, preventive care, specialist referrals'
    }
    return descriptions.get(agent_key, 'Medical analysis and insights')

@app.route('/analysis')
def analysis_page():
    """Multi-agent analysis page"""
    if not gemini_model:
        return render_template('error.html', 
                             error_message="Gemini API not configured. Please add your API key to the .env file.")
    
    return render_template('analysis.html')

@app.route('/analyze', methods=['POST'])
def analyze_report():
    """Analyze medical report using multi-agent system"""
    if not agent_router:
        return jsonify({'error': 'AI system not available. Please check your API configuration.'}), 500
    
    try:
        # Handle both JSON and form data
        if request.content_type and 'application/json' in request.content_type:
            # JSON request (manual text input)
            data = request.json
            report_text = data.get('report_text', '').strip()
            symptoms = data.get('symptoms', '')
            medical_history = data.get('medical_history', '')
            file_info = None
        else:
            # Form request (file upload)
            report_text = request.form.get('report_text', '').strip()
            symptoms = request.form.get('symptoms', '')
            medical_history = request.form.get('medical_history', '')
            
            # Check if file was uploaded
            if 'report_file' in request.files:
                file = request.files['report_file']
                if file and file.filename:
                    try:
                        extracted_text = process_uploaded_file(file)
                        report_text = extracted_text
                        file_info = {
                            'filename': secure_filename(file.filename),
                            'size': len(extracted_text),
                            'type': file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'unknown'
                        }
                    except Exception as e:
                        return jsonify({'error': f'File processing failed: {str(e)}'}), 400
        
        if not report_text:
            return jsonify({'error': 'Please provide a medical report (either by uploading a file or entering text manually).'}), 400
        
        # Get multi-agent analysis
        analysis_result = agent_router.get_multi_agent_analysis(report_text, symptoms, medical_history)
        
        # Format response
        primary_agent = analysis_result['primary_agent']
        response_data = {
            'primary_agent': {
                'key': primary_agent['key'],
                'name': primary_agent['agent'].specialty,
                'icon': primary_agent['agent'].specialty_icon,
                'analysis': primary_agent['analysis'],
                'confidence': primary_agent['confidence']
            },
            'secondary_agents': [],
            'routing_info': analysis_result['all_confidence_scores'],
            'file_info': file_info if 'file_info' in locals() else None
        }
        
        # Add secondary agents
        for agent_key, agent_data in analysis_result['secondary_agents'].items():
            response_data['secondary_agents'].append({
                'key': agent_key,
                'name': agent_data['agent'].specialty,
                'icon': agent_data['agent'].specialty_icon,
                'analysis': agent_data['analysis'],
                'confidence': agent_data['confidence']
            })
        
        # Save to session history
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'report_text': report_text[:200] + '...' if len(report_text) > 200 else report_text,
            'symptoms': symptoms,
            'medical_history': medical_history,
            'primary_agent': primary_agent['agent'].specialty,
            'primary_analysis': primary_agent['analysis'],
            'agent_confidence': primary_agent['confidence'],
            'secondary_agents': list(analysis_result['secondary_agents'].keys())
        }
        
        session['diagnosis_history'].append(history_entry)
        session.modified = True
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/chat')
def chat_page():
    """Medical chat page"""
    if not gemini_model:
        return render_template('error.html', 
                             error_message="Gemini API not configured. Please add your API key to the .env file.")
    
    return render_template('chat.html', chat_history=session['chat_history'])

@app.route('/send_message', methods=['POST'])
def send_message():
    """Send message to AI chat"""
    if not agent_router:
        return jsonify({'error': 'AI system not available.'}), 500
    
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Please enter a message.'}), 400
        
        # Add user message to history
        session['chat_history'].append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Get AI response using general medicine agent for chat
        general_agent = agent_router.agents['general']
        
        # Create a chat-specific prompt for general medical questions
        chat_prompt = f"""
        You are a friendly medical education AI assistant. Provide helpful but BRIEF medical information.
        
        USER MESSAGE: {user_message}
        
        INSTRUCTIONS:
        - Keep responses SHORT - maximum 3-4 sentences for simple questions
        - If greeting (like "hey", "hello", "hi"), respond warmly in 1-2 sentences
        - For medical questions, give essential information only
        - Use simple, clear language without any symbols (* # ** ## etc.)
        - Be conversational but professional
        - Always end with: "Consult a healthcare professional for personalized advice."
        
        RESPONSE LENGTH LIMITS:
        - Greetings: 1-2 sentences maximum
        - Simple questions: 3-4 sentences maximum  
        - Complex topics: 5-6 sentences maximum (rarely needed)
        - NEVER write long paragraphs or detailed explanations
        
        FORMATTING RULES:
        - NO symbols, markdown, or special formatting
        - Write in short, natural sentences
        - Use simple line breaks between points if needed
        - Keep it conversational and brief
        
        EXAMPLE BRIEF RESPONSE:
        A fever is your body fighting infection by raising temperature above 98.6°F. Most fevers from colds/flu are helpful, not harmful. See a doctor if fever reaches 103°F, lasts over 3 days, or causes severe symptoms. Rest, drink fluids, and use acetaminophen as directed. Consult a healthcare professional for personalized advice.
        
        CRITICAL: Keep responses SHORT and use only plain text.
        """
        
        ai_response = make_api_call_with_retry(general_agent.model, chat_prompt)
        
        # Add AI response to history
        session['chat_history'].append({
            'role': 'assistant',
            'content': ai_response,
            'timestamp': datetime.now().isoformat()
        })
        
        session.modified = True
        
        return jsonify({
            'user_message': user_message,
            'ai_response': ai_response
        })
        
    except Exception as e:
        return jsonify({'error': f'Chat failed: {str(e)}'}), 500

@app.route('/history')
def history_page():
    """Diagnosis history page"""
    # Calculate statistics
    history = session['diagnosis_history']
    stats = {
        'total_analyses': len(history),
        'this_month': 0,
        'last_analysis': None,
        'agent_usage': {}
    }
    
    if history:
        # Calculate this month's analyses
        current_month = datetime.now().month
        current_year = datetime.now().year
        
        for entry in history:
            entry_date = datetime.fromisoformat(entry['timestamp'])
            if entry_date.month == current_month and entry_date.year == current_year:
                stats['this_month'] += 1
            
            # Track agent usage
            agent = entry.get('primary_agent', 'Unknown')
            stats['agent_usage'][agent] = stats['agent_usage'].get(agent, 0) + 1
        
        # Last analysis date
        stats['last_analysis'] = datetime.fromisoformat(history[-1]['timestamp']).strftime('%Y-%m-%d')
    
    return render_template('history.html', 
                         history=history, 
                         stats=stats)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear diagnosis history"""
    session['diagnosis_history'] = []
    session.modified = True
    return jsonify({'success': True})

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    """Clear chat history"""
    session['chat_history'] = []
    session.modified = True
    return jsonify({'success': True})

@app.route('/export_history')
def export_history():
    """Export diagnosis history as JSON"""
    from flask import make_response
    
    response_data = {
        'export_date': datetime.now().isoformat(),
        'total_analyses': len(session['diagnosis_history']),
        'diagnosis_history': session['diagnosis_history']
    }
    
    response = make_response(json.dumps(response_data, indent=2))
    response.headers['Content-Type'] = 'application/json'
    response.headers['Content-Disposition'] = f'attachment; filename=medical_analysis_history_{datetime.now().strftime("%Y%m%d")}.json'
    
    return response

@app.route('/api/status')
def api_status():
    """API endpoint for system status"""
    return jsonify({
        'gemini_configured': gemini_model is not None,
        'agents_ready': agent_router is not None,
        'total_analyses': len(session['diagnosis_history']),
        'chat_sessions': len(session['chat_history']),
        'available_agents': len(agent_router.agents) if agent_router else 0
    })

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', 
                         error_message="Page not found. Please check the URL and try again."), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', 
                         error_message="Internal server error. Please try again later."), 500

if __name__ == '__main__':
    # Initialize the system
    gemini_model = configure_gemini()
    if gemini_model:
        agent_router = MedicalAgentRouter(gemini_model)
        print("🤖 Multi-Agent Medical Diagnostics System initialized successfully!")
    else:
        print("⚠️  Gemini API not configured. Please add GEMINI_API_KEY to .env file.")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
