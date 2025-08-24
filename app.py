import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
from PIL import Image
import io
import time
import random
from config import Config
from agents import MedicalAgentRouter, make_api_call_with_retry

# Load environment variables (ignore if file doesn't exist)
try:
    load_dotenv()
except (FileNotFoundError, UnicodeDecodeError):
    print("Warning: .env file not found or corrupted in app.py. Using default configuration.")

# Configure page
st.set_page_config(
    page_title="AI Medical Diagnostics Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure Gemini API
def configure_gemini():
    """Configure Google Gemini AI"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY not found in environment variables!")
        st.info("Please add your Gemini API key to the .env file")
        return None
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(Config.GEMINI_MODEL)



def optimize_prompt(prompt, max_tokens=2000):
    """
    Optimize prompt to reduce token usage while maintaining quality
    """
    # If prompt is too long, truncate while keeping important parts
    if len(prompt) > max_tokens:
        # Keep system prompt and truncate user content
        lines = prompt.split('\n')
        system_lines = []
        content_lines = []
        
        in_system = True
        for line in lines:
            if "MEDICAL REPORT:" in line or "Please analyze" in line:
                in_system = False
            
            if in_system:
                system_lines.append(line)
            else:
                content_lines.append(line)
        
        # Keep system prompt and truncate content if needed
        system_text = '\n'.join(system_lines)
        content_text = '\n'.join(content_lines)
        
        if len(system_text + content_text) > max_tokens:
            # Truncate content while keeping structure
            content_text = content_text[:max_tokens - len(system_text) - 100] + "\n\n[Content truncated to reduce token usage]"
        
        return system_text + '\n' + content_text
    
    return prompt

def make_api_call_with_retry(model, prompt, max_retries=3, base_delay=1):
    """
    Make API call with exponential backoff retry logic for rate limiting
    """
    # Optimize prompt to reduce token usage
    optimized_prompt = optimize_prompt(prompt)
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(optimized_prompt)
            return response.text
        except Exception as e:
            error_str = str(e)
            
            # Check for quota/rate limit errors
            if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                if attempt < max_retries - 1:
                    # Extract retry delay from error message if available
                    retry_delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    
                    # Check if error message contains suggested delay
                    if "retry_delay" in error_str and "seconds" in error_str:
                        try:
                            # Extract the suggested delay from error message
                            import re
                            delay_match = re.search(r'retry_delay.*?seconds:\s*(\d+)', error_str)
                            if delay_match:
                                suggested_delay = int(delay_match.group(1))
                                retry_delay = max(retry_delay, suggested_delay)
                        except:
                            pass
                    
                    st.warning(f"⏳ Processing your request... Please wait {retry_delay:.1f} seconds (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                else:
                    return """
                    🚫 **Service Temporarily Unavailable**
                    
                    The AI service is currently experiencing high demand.
                    
                    **What you can do:**
                    
                    ⏳ **Wait a moment** - The service will be available again shortly
                    
                    📝 **Try shorter requests** - Use more concise text for better results
                    
                    🔄 **Try again** - The system will automatically retry your request
                    
                    We apologize for the inconvenience. Please try again in a few minutes.
                    """
            else:
                # Other types of errors
                return "Sorry, there was an issue processing your request. Please try again."
    
    return "Maximum retry attempts exceeded. Please try again later."

class MultiAgentMedicalSystem:
    """Multi-agent medical diagnostic system with specialized agents"""
    
    def __init__(self, model):
        self.model = model
        self.agent_router = MedicalAgentRouter(model)
        self.system_prompt = """
        You are an AI Medical Assistant providing educational information about health topics.
        
        IMPORTANT DISCLAIMERS:
        - You are NOT a replacement for professional medical diagnosis
        - Always recommend consulting with healthcare professionals
        - Your insights are for informational purposes only
        - Do not provide specific treatment recommendations
        """
    
    def analyze_report(self, report_text, symptoms="", medical_history=""):
        """Analyze medical report using specialized agents"""
        return self.agent_router.get_multi_agent_analysis(report_text, symptoms, medical_history)
    
    def get_medical_insights(self, query):
        """Get general medical insights for health queries"""
        prompt = f"""
        You are a friendly medical education AI assistant. Provide helpful but BRIEF medical information.
        
        USER MESSAGE: {query}
        
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
        
        return make_api_call_with_retry(self.model, prompt)

def main():
    # Initialize session state first
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'diagnosis_history' not in st.session_state:
        st.session_state.diagnosis_history = []
    
    st.title("🤖 Multi-Agent AI Medical Diagnostics")
    st.markdown("### Specialized AI Medical Experts Working Together")
    
    # Show active agents status
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🫀 Cardiology", "Active", delta="Ready")
    with col2:
        st.metric("🧠 Psychology", "Active", delta="Ready")
    with col3:
        st.metric("🫁 Pulmonology", "Active", delta="Ready")
    with col4:
        st.metric("🩺 General Med", "Active", delta="Ready")
    

    
    # Configure Gemini
    model = configure_gemini()
    if not model:
        return
    
    # Initialize multi-agent diagnostic system
    diagnostic_system = MultiAgentMedicalSystem(model)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings")
        
        # Service Status
        if model:
            st.success("✅ Multi-Agent System Ready")
            st.markdown("""
            **Active Specialists:**
            - 🫀 Cardiologist Agent
            - 🧠 Psychology Agent  
            - 🫁 Pulmonology Agent
            - 🩺 General Medicine Agent
            """)
        else:
            st.error("❌ AI System Unavailable")
        
        st.markdown("---")
        
        # Quick Actions
        st.subheader("🔧 Quick Actions")
        
        if st.button("🗑️ Clear Chat History", help="Start fresh conversation"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigate to:",
            ["🏠 Home", "🤖 Multi-Agent Analysis", "💬 Medical Chat", "📊 Diagnosis History"]
        )
        
        st.markdown("---")
        
        # Important Notice
        st.warning("""
        ⚠️ **Important Notice**
        
        This assistant provides educational information only. 
        Always consult healthcare professionals for medical decisions.
        """)
    
    # Main content based on navigation
    if page == "🏠 Home":
        show_home_page()
    elif page == "🤖 Multi-Agent Analysis":
        show_report_analysis(diagnostic_system)
    elif page == "💬 Medical Chat":
        show_medical_chat(diagnostic_system)
    elif page == "📊 Diagnosis History":
        show_diagnosis_history()

def show_home_page():
    """Display home page with features overview"""
    st.markdown("""
    ## Welcome to Multi-Agent AI Medical Diagnostics Assistant
    
    This intelligent system uses **specialized AI medical agents** to analyze your reports and route them to the most appropriate medical expertise using advanced AI technology.
    
    ### 🤖 Our AI Medical Specialists:
    """)
    
    # Display agent cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **🫀 Cardiologist Agent**
        - Heart conditions
        - Blood pressure
        - Cardiovascular health
        - ECG analysis
        """)
    
    with col2:
        st.markdown("""
        **🧠 Psychology Agent**
        - Mental health
        - Depression & anxiety
        - Stress management
        - Sleep disorders
        """)
    
    with col3:
        st.markdown("""
        **🫁 Pulmonology Agent**
        - Lung conditions
        - Breathing issues
        - Respiratory health
        - Chest imaging
        """)
    
    with col4:
        st.markdown("""
        **🩺 General Medicine**
        - Overall health
        - Lab results
        - Preventive care
        - Specialist referrals
        """)
    
    st.markdown("""
    ### 🚀 How It Works:
    
    1. **📄 Upload Your Report** - Paste or upload your medical report
    2. **🎯 Smart Routing** - Our AI analyzes content and routes to the best specialist
    3. **🤖 Expert Analysis** - Specialized agents provide targeted insights
    4. **📊 Multi-Perspective** - Get opinions from multiple relevant specialists
    
    ### 🚀 Features:
    
    #### 📋 Multi-Agent Report Analysis
    - **Intelligent Routing**: Automatically selects the most relevant medical specialist
    - **Specialized Insights**: Each agent focuses on their area of expertise
    - **Multiple Perspectives**: Get secondary opinions from other relevant specialists
    - **Confidence Scoring**: See how confident each agent is about the analysis
    
    #### 💬 Interactive Medical Chat
    - Ask health-related questions to our AI specialists
    - Get educational information about symptoms and conditions
    - Receive guidance on when to seek professional care
    
    #### 📊 Enhanced Diagnosis History
    - Track which specialists analyzed your reports
    - Review confidence scores and routing decisions
    - Export comprehensive findings for healthcare providers
    
    ### 🛡️ Safety & Privacy:
    - All analysis is for **educational purposes only**
    - Your data is processed securely
    - Always consult healthcare professionals for medical decisions
    - No medical advice or treatment recommendations provided
    """)
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔍 Analyses Completed", len(st.session_state.diagnosis_history))
    
    with col2:
        st.metric("💬 Chat Sessions", len(st.session_state.chat_history))
    
    with col3:
        st.metric("🤖 AI Assistant", "Active")

def show_report_analysis(diagnostic_system):
    """Display report analysis interface with multi-agent system"""
    st.header("📋 Multi-Agent Medical Report Analysis")
    st.markdown("*Our AI specialists will analyze your report and route it to the most appropriate medical expert*")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📄 Upload Report")
        
        # Text input for report
        report_text = st.text_area(
            "Paste your medical report here:",
            height=200,
            placeholder="Enter your medical report text, lab results, or diagnostic findings here..."
        )
        
        # File upload option
        uploaded_file = st.file_uploader(
            "Or upload a text file:",
            type=['txt', 'pdf'],
            help="Upload a text or PDF file containing your medical report"
        )
        
        if uploaded_file is not None:
            if uploaded_file.type == "text/plain":
                report_text = str(uploaded_file.read(), "utf-8")
                st.success("✅ File uploaded successfully!")
            else:
                st.warning("PDF processing not implemented yet. Please use text files or copy-paste.")
    
    with col2:
        st.subheader("ℹ️ Additional Information")
        
        symptoms = st.text_area(
            "Current Symptoms:",
            height=100,
            placeholder="Describe any current symptoms you're experiencing..."
        )
        
        medical_history = st.text_area(
            "Medical History:",
            height=100,
            placeholder="Relevant medical history, medications, allergies..."
        )
    
    # Analysis button
    if st.button("🔍 Analyze with AI Specialists", type="primary", use_container_width=True):
        if report_text.strip():
            with st.spinner("🤖 Routing to appropriate medical specialists..."):
                # Get multi-agent analysis
                analysis_result = diagnostic_system.analyze_report(report_text, symptoms, medical_history)
                
                # Display agent routing information
                st.markdown("---")
                st.subheader("🎯 AI Agent Routing")
                
                primary_agent = analysis_result["primary_agent"]
                agent_info = primary_agent["agent"]
                
                # Show primary agent
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    st.markdown(f"### {agent_info.specialty_icon}")
                with col2:
                    st.markdown(f"**Primary Specialist: {agent_info.specialty}**")
                    st.progress(primary_agent["confidence"])
                with col3:
                    st.metric("Confidence", f"{primary_agent['confidence']:.0%}")
                
                # Show secondary agents if any
                if analysis_result["secondary_agents"]:
                    st.markdown("**Additional Consultations:**")
                    for agent_key, agent_data in analysis_result["secondary_agents"].items():
                        agent_obj = agent_data["agent"]
                        confidence = agent_data["confidence"]
                        st.markdown(f"• {agent_obj.specialty_icon} {agent_obj.specialty} ({confidence:.0%} relevance)")
                
                # Display primary analysis
                st.markdown("---")
                st.subheader(f"{agent_info.specialty_icon} Primary Analysis - {agent_info.specialty}")
                st.markdown(primary_agent["analysis"])
                
                # Display secondary analyses if any
                if analysis_result["secondary_agents"]:
                    st.markdown("---")
                    st.subheader("🔍 Additional Specialist Insights")
                    
                    for agent_key, agent_data in analysis_result["secondary_agents"].items():
                        agent_obj = agent_data["agent"]
                        with st.expander(f"{agent_obj.specialty_icon} {agent_obj.specialty} Perspective"):
                            st.markdown(agent_data["analysis"])
                
                # Save to history with agent information
                history_entry = {
                    "timestamp": pd.Timestamp.now(),
                    "report_text": report_text[:200] + "..." if len(report_text) > 200 else report_text,
                    "symptoms": symptoms,
                    "medical_history": medical_history,
                    "primary_agent": agent_info.specialty,
                    "primary_analysis": primary_agent["analysis"],
                    "agent_confidence": primary_agent["confidence"],
                    "secondary_agents": list(analysis_result["secondary_agents"].keys()) if analysis_result["secondary_agents"] else []
                }
                
                st.session_state.diagnosis_history.append(history_entry)
                
                st.success(f"✅ Analysis completed by {agent_info.specialty} specialist and saved to history!")
        else:
            st.error("⚠️ Please enter a medical report to analyze.")

def show_medical_chat(diagnostic_system):
    """Display medical chat interface"""
    st.header("💬 AI Medical Chat Assistant")
    st.markdown("*Ask questions and get educational information from our AI medical specialists*")
    
    # Chat history display
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask me about symptoms, conditions, or health concerns...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get AI response
        with st.spinner("💭 Analyzing your question..."):
            response = diagnostic_system.get_medical_insights(user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Rerun to update chat display
        st.rerun()
    
    # Quick action buttons
    st.markdown("---")
    st.subheader("🚀 Quick Questions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🤒 About Fever"):
            user_input = "Tell me about fever symptoms and when to seek medical attention"
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            response = diagnostic_system.get_medical_insights(user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col2:
        if st.button("🫀 Heart Health"):
            user_input = "What are signs of heart problems I should watch for?"
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            response = diagnostic_system.get_medical_insights(user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col3:
        if st.button("🧠 Mental Health"):
            user_input = "How can I recognize signs of depression or anxiety?"
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            response = diagnostic_system.get_medical_insights(user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

def show_diagnosis_history():
    """Display diagnosis history and analytics with agent information"""
    st.header("📊 Multi-Agent Diagnosis History")
    
    if not st.session_state.diagnosis_history:
        st.info("📝 No analysis history yet. Start by analyzing a medical report with our AI specialists!")
        return
    
    # Create DataFrame from history
    df = pd.DataFrame(st.session_state.diagnosis_history)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyses", len(df))
    
    with col2:
        st.metric("This Month", len(df[df['timestamp'].dt.month == pd.Timestamp.now().month]))
    
    with col3:
        if len(df) > 0:
            st.metric("Last Analysis", df['timestamp'].max().strftime("%Y-%m-%d"))
    
    with col4:
        # Most used agent
        if 'primary_agent' in df.columns and len(df) > 0:
            most_used_agent = df['primary_agent'].mode().iloc[0] if not df['primary_agent'].mode().empty else "N/A"
            st.metric("Top Specialist", most_used_agent)
        else:
            st.metric("🤖 Active Agents", "4 Specialists")
    
    # Agent usage statistics
    if 'primary_agent' in df.columns and len(df) > 0:
        st.subheader("🤖 AI Specialist Usage")
        agent_counts = df['primary_agent'].value_counts()
        
        # Create agent usage chart
        fig = px.pie(values=agent_counts.values, names=agent_counts.index, 
                    title="Distribution of Cases by Medical Specialist")
        st.plotly_chart(fig, use_container_width=True)
    
    # Analysis timeline
    if len(df) > 1:
        st.subheader("📈 Analysis Timeline")
        df['date'] = df['timestamp'].dt.date
        daily_counts = df.groupby('date').size().reset_index(name='count')
        
        fig = px.line(daily_counts, x='date', y='count', 
                     title="Daily Analysis Count")
        st.plotly_chart(fig, use_container_width=True)
    
    # History table
    st.subheader("📋 Analysis History")
    
    for i, row in df.iterrows():
        # Determine agent icon
        agent_icon = "🩺"  # default
        if 'primary_agent' in row:
            if "Cardiology" in str(row['primary_agent']):
                agent_icon = "🫀"
            elif "Psychology" in str(row['primary_agent']):
                agent_icon = "🧠"
            elif "Pulmonology" in str(row['primary_agent']):
                agent_icon = "🫁"
        
        title = f"{agent_icon} Analysis {i+1} - {row['timestamp'].strftime('%Y-%m-%d %H:%M')}"
        if 'primary_agent' in row:
            title += f" ({row['primary_agent']})"
        
        with st.expander(title):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**Report Summary:**")
                st.write(row['report_text'])
                
                if 'symptoms' in row and row['symptoms']:
                    st.write("**Symptoms:**")
                    st.write(row['symptoms'])
                elif row.get('symptoms'):  # Fallback
                    st.write("**Symptoms:**")
                    st.write(row['symptoms'])
                
                if 'medical_history' in row and row['medical_history']:
                    st.write("**Medical History:**")
                    st.write(row['medical_history'])
                elif row.get('medical_history'):  # Fallback
                    st.write("**Medical History:**")
                    st.write(row['medical_history'])
                
                # Show agent information if available
                if 'primary_agent' in row:
                    st.write("**Primary Specialist:**")
                    st.write(f"{agent_icon} {row['primary_agent']}")
                    
                    if 'agent_confidence' in row:
                        st.write("**Confidence Score:**")
                        st.progress(row['agent_confidence'])
                        st.write(f"{row['agent_confidence']:.0%}")
                    
                    if 'secondary_agents' in row and row['secondary_agents']:
                        st.write("**Additional Consultations:**")
                        for agent in row['secondary_agents']:
                            st.write(f"• {agent}")
            
            with col2:
                if 'primary_analysis' in row:
                    st.write(f"**{agent_icon} Primary Analysis:**")
                    st.write(row['primary_analysis'])
                elif 'analysis' in row:  # Fallback for old format
                    st.write("**AI Analysis:**")
                    st.write(row['analysis'])
    
    # Export functionality
    if st.button("📥 Export History", type="secondary"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"multi_agent_medical_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
