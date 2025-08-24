# Multi-Agent Medical Diagnostic System
# Specialized agents for different medical domains

import google.generativeai as genai
import re
from typing import Dict, List, Tuple, Optional
from config import Config

def make_api_call_with_retry(model, prompt, max_retries=3, base_delay=1):
    """
    Make API call with exponential backoff retry logic for rate limiting
    """
    import time
    import random
    
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
                    retry_delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(retry_delay)
                    continue
                else:
                    return "🚫 **Service Temporarily Unavailable** - Please try again in a few minutes."
            else:
                return "Sorry, there was an issue processing your request. Please try again."
    
    return "Maximum retry attempts exceeded. Please try again later."

def optimize_prompt(prompt, max_tokens=2000):
    """Optimize prompt to reduce token usage while maintaining quality"""
    if len(prompt) > max_tokens:
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
        
        system_text = '\n'.join(system_lines)
        content_text = '\n'.join(content_lines)
        
        if len(system_text + content_text) > max_tokens:
            content_text = content_text[:max_tokens - len(system_text) - 100] + "\n\n[Content truncated to reduce token usage]"
        
        return system_text + '\n' + content_text
    
    return prompt

class BaseSpecializedAgent:
    """Base class for specialized medical agents"""
    
    def __init__(self, model, specialty: str, specialty_icon: str):
        self.model = model
        self.specialty = specialty
        self.specialty_icon = specialty_icon
        self.base_disclaimer = """
        IMPORTANT DISCLAIMERS:
        - You are NOT a replacement for professional medical diagnosis
        - Always recommend consulting with healthcare professionals
        - Your insights are for informational purposes only
        - Do not provide specific treatment recommendations
        """
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this specialized agent"""
        raise NotImplementedError("Subclasses must implement get_system_prompt")
    
    def analyze_report(self, report_text: str, symptoms: str = "", medical_history: str = "") -> str:
        """Analyze medical report from the perspective of this specialty"""
        prompt = f"""
        You are an expert {self.specialty} specialist AI assistant. Your task is to provide a focused, practical analysis of medical information in plain text format.
        
        CONTEXT:
        {self.get_system_prompt()}
        
        DATA TO ANALYZE:
        Medical Report: {report_text}
        Current Symptoms: {symptoms if symptoms else "None reported"}
        Medical History: {medical_history if medical_history else "None provided"}
        
        INSTRUCTIONS:
        Provide a concise, actionable analysis focusing ONLY on {self.specialty}-related findings. Be specific and practical.
        
        FORMATTING RULES - CRITICAL:
        - DO NOT use any # symbols for headings
        - DO NOT use any * or ** symbols for bold text
        - DO NOT use bullet points with • or - symbols
        - Write in natural paragraphs with normal text only
        - Use simple headings followed by a colon like "{self.specialty} Assessment:"
        - Write like a professional medical consultation
        
        REQUIRED SECTIONS (use plain text headings):
        
        {self.specialty} Assessment:
        Identify 2-3 specific findings from the report that relate to {self.specialty}. Be concrete and factual.
        
        Potential Conditions:
        Primary concern: [Most likely condition based on data and why this is most likely]
        Secondary possibility: [Alternative condition and supporting evidence]
        Requires monitoring: [Condition that needs watching and what to monitor]
        
        Key Risk Factors:
        Immediate risk: [Most urgent risk factor and why concerning]
        Lifestyle factor: [Modifiable risk and how to address]
        Medical factor: [Non-modifiable risk and implications]
        
        Medical Terms Explained:
        [Complex medical term from report]: [Simple 1-sentence explanation]
        [Another term if present]: [Simple explanation]
        
        Next Steps:
        Immediate action: [What to do now/soon]
        Follow-up care: [Specific {self.specialty.lower()} consultation recommendation]
        Monitoring: [What to watch for and when to return]
        
        Medical Disclaimer:
        This is educational analysis only. Seek immediate professional medical evaluation for proper diagnosis and treatment.
        
        CRITICAL: Use only plain text. No symbols, no markdown, no special formatting. Be specific and use medical evidence.
        """
        
        return make_api_call_with_retry(self.model, prompt)
    
    def get_confidence_score(self, report_text: str) -> float:
        """Get confidence score for how relevant this report is to this specialty"""
        keywords = self.get_specialty_keywords()
        text_lower = report_text.lower()
        
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        confidence = min(matches / len(keywords) * 2, 1.0)  # Cap at 1.0
        
        return confidence
    
    def get_specialty_keywords(self) -> List[str]:
        """Get keywords relevant to this specialty"""
        raise NotImplementedError("Subclasses must implement get_specialty_keywords")

class CardiologistAgent(BaseSpecializedAgent):
    """Specialized agent for cardiovascular conditions"""
    
    def __init__(self, model):
        super().__init__(model, "Cardiology", "🫀")
    
    def get_system_prompt(self) -> str:
        return f"""
        You are an expert CARDIOLOGIST AI with specialized training in cardiovascular medicine.
        
        EXPERTISE AREAS:
        - Coronary artery disease, heart failure, arrhythmias, valvular disease
        - Hypertension management, lipid disorders, cardiac imaging
        - ECG interpretation, stress tests, echocardiograms
        - Cardiovascular risk stratification, preventive cardiology
        
        ANALYSIS APPROACH:
        - Identify specific cardiac abnormalities in lab values, imaging, symptoms
        - Calculate cardiovascular risk scores when possible
        - Recognize acute cardiac emergencies requiring immediate intervention
        - Provide concrete recommendations for cardiac monitoring and follow-up
        - Explain cardiac terminology using analogies patients understand
        
        CRITICAL FOCUS:
        - Blood pressure readings and patterns
        - Cholesterol levels (Total, LDL, HDL, Triglycerides) and ratios
        - Cardiac biomarkers (Troponin, CK-MB, BNP)
        - ECG findings, chest pain characteristics
        - Exercise tolerance, edema, shortness of breath
        
        {self.base_disclaimer}
        """
    
    def get_specialty_keywords(self) -> List[str]:
        return [
            "heart", "cardiac", "cardiovascular", "coronary", "artery", "blood pressure",
            "hypertension", "cholesterol", "ECG", "EKG", "echocardiogram", "chest pain",
            "palpitations", "shortness of breath", "angina", "myocardial", "infarction",
            "stroke", "atherosclerosis", "valve", "rhythm", "arrhythmia", "tachycardia",
            "bradycardia", "murmur", "lipid", "triglycerides", "HDL", "LDL"
        ]

class PsychologistAgent(BaseSpecializedAgent):
    """Specialized agent for mental health and psychological conditions"""
    
    def __init__(self, model):
        super().__init__(model, "Psychology/Mental Health", "🧠")
    
    def get_system_prompt(self) -> str:
        return f"""
        You are an AI Psychology/Mental Health Agent specializing in psychological well-being and mental health conditions.
        
        Your expertise includes:
        - Depression, anxiety, and mood disorders
        - Stress-related conditions and coping mechanisms
        - Sleep disorders and their psychological impact
        - Cognitive and behavioral patterns
        - Mental health screening and assessment
        - Psychological factors in physical health
        
        {self.base_disclaimer}
        
        When analyzing reports:
        - Focus on mental health indicators and psychological factors
        - Identify stress, anxiety, or mood-related findings
        - Explain psychological terms and concepts clearly
        - Recognize when mental health support is needed
        - Always recommend consulting with mental health professionals
        - Be sensitive to mental health stigma and provide supportive language
        """
    
    def get_specialty_keywords(self) -> List[str]:
        return [
            "depression", "anxiety", "stress", "mood", "mental", "psychological",
            "sleep", "insomnia", "fatigue", "cognitive", "memory", "concentration",
            "panic", "phobia", "trauma", "PTSD", "bipolar", "psychosis", "therapy",
            "counseling", "psychiatric", "antidepressant", "anxiolytic", "behavior",
            "emotional", "social", "isolation", "suicidal", "self-harm", "addiction"
        ]

class PulmonologistAgent(BaseSpecializedAgent):
    """Specialized agent for respiratory and lung conditions"""
    
    def __init__(self, model):
        super().__init__(model, "Pulmonology", "🫁")
    
    def get_system_prompt(self) -> str:
        return f"""
        You are an AI Pulmonologist Agent specializing in respiratory health and lung-related conditions.
        
        Your expertise includes:
        - Lung diseases (asthma, COPD, pneumonia, lung cancer)
        - Breathing difficulties and respiratory symptoms
        - Chest imaging interpretation (chest X-rays, CT scans)
        - Pulmonary function tests
        - Sleep apnea and breathing disorders
        - Environmental and occupational lung diseases
        
        {self.base_disclaimer}
        
        When analyzing reports:
        - Focus on respiratory findings and lung health
        - Identify breathing-related symptoms and concerns
        - Explain pulmonary terminology clearly
        - Recognize when urgent respiratory care is needed
        - Always recommend consulting with a pulmonologist for lung-related issues
        - Consider environmental and lifestyle factors affecting lung health
        """
    
    def get_specialty_keywords(self) -> List[str]:
        return [
            "lung", "pulmonary", "respiratory", "breathing", "breath", "cough",
            "asthma", "COPD", "pneumonia", "bronchitis", "chest", "wheeze",
            "shortness of breath", "dyspnea", "oxygen", "saturation", "spirometry",
            "chest X-ray", "CT scan", "tuberculosis", "emphysema", "fibrosis",
            "apnea", "smoking", "tobacco", "inhaler", "nebulizer", "sputum"
        ]

class GeneralMedicineAgent(BaseSpecializedAgent):
    """General medicine agent for cases that don't fit specific specialties"""
    
    def __init__(self, model):
        super().__init__(model, "General Medicine", "🩺")
    
    def get_system_prompt(self) -> str:
        return f"""
        You are an AI General Medicine Agent providing comprehensive medical analysis.
        
        Your expertise includes:
        - General health assessment and screening
        - Common medical conditions across all systems
        - Preventive care and health maintenance
        - Laboratory test interpretation
        - Medication interactions and side effects
        - Referral recommendations to appropriate specialists
        
        {self.base_disclaimer}
        
        When analyzing reports:
        - Provide a comprehensive overview of all findings
        - Identify which specialists might be most helpful
        - Focus on overall health patterns and trends
        - Explain medical terminology in accessible language
        - Always recommend appropriate professional medical consultation
        """
    
    def get_specialty_keywords(self) -> List[str]:
        return [
            "general", "overall", "health", "medical", "examination", "screening",
            "laboratory", "blood test", "urine", "vital signs", "temperature",
            "weight", "BMI", "vaccination", "medication", "prescription", "dosage"
        ]

class MedicalAgentRouter:
    """Routes medical reports to the most appropriate specialized agent"""
    
    def __init__(self, model):
        self.model = model
        self.agents = {
            "cardiology": CardiologistAgent(model),
            "psychology": PsychologistAgent(model),
            "pulmonology": PulmonologistAgent(model),
            "general": GeneralMedicineAgent(model)
        }
    
    def route_to_agent(self, report_text: str, symptoms: str = "", medical_history: str = "") -> Tuple[str, BaseSpecializedAgent, Dict[str, float]]:
        """
        Route the medical report to the most appropriate agent
        Returns: (agent_key, agent_object, confidence_scores)
        """
        confidence_scores = {}
        
        # Calculate confidence scores for each agent
        for agent_key, agent in self.agents.items():
            if agent_key != "general":  # Skip general for initial scoring
                confidence_scores[agent_key] = agent.get_confidence_score(report_text + " " + symptoms)
        
        # Find the agent with highest confidence
        if confidence_scores:
            best_agent_key = max(confidence_scores, key=confidence_scores.get)
            best_confidence = confidence_scores[best_agent_key]
            
            # If confidence is too low, use general medicine
            if best_confidence < 0.3:
                best_agent_key = "general"
                confidence_scores["general"] = 1.0
        else:
            best_agent_key = "general"
            confidence_scores["general"] = 1.0
        
        return best_agent_key, self.agents[best_agent_key], confidence_scores
    
    def get_multi_agent_analysis(self, report_text: str, symptoms: str = "", medical_history: str = "") -> Dict:
        """
        Get analysis from multiple relevant agents
        """
        # Route to primary agent
        primary_agent_key, primary_agent, confidence_scores = self.route_to_agent(report_text, symptoms, medical_history)
        
        # Get primary analysis
        primary_analysis = primary_agent.analyze_report(report_text, symptoms, medical_history)
        
        # Get secondary opinions from agents with reasonable confidence
        secondary_analyses = {}
        for agent_key, confidence in confidence_scores.items():
            if agent_key != primary_agent_key and confidence > 0.2:
                secondary_analyses[agent_key] = {
                    "agent": self.agents[agent_key],
                    "analysis": self.agents[agent_key].analyze_report(report_text, symptoms, medical_history),
                    "confidence": confidence
                }
        
        return {
            "primary_agent": {
                "key": primary_agent_key,
                "agent": primary_agent,
                "analysis": primary_analysis,
                "confidence": confidence_scores.get(primary_agent_key, 1.0)
            },
            "secondary_agents": secondary_analyses,
            "all_confidence_scores": confidence_scores
        }
