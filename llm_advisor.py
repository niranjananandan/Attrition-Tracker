import os
# pyrefly: ignore [missing-import]
import google.generativeai as genai

def generate_action_plan(risk_factors, retention_factors):
    """
    Calls the Gemini API to generate an HR action plan based on the risk and retention factors.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ **Configuration Error**: Gemini API key is missing. Please set `GEMINI_API_KEY` in your `.env` file."
        
    if not risk_factors and not retention_factors:
        return "No risk or retention factors provided to generate a strategy."
        
    try:
        genai.configure(api_key=api_key)
        prompt = f"""You are an Expert HR Consultant. An employee has these Risk Factors pushing them to leave: {risk_factors}. They have these Retention Factors keeping them: {retention_factors}. Provide exactly 3 highly professional, short, and actionable retention strategies to save this employee. Format as a clean markdown list without extra fluff."""
        
        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(prompt)
        except Exception as e:
            # Fallback if gemini-1.5-flash-latest is not found in the environment
            if "not found" in str(e).lower() or "404" in str(e):
                model = genai.GenerativeModel('gemini-flash-latest')
                response = model.generate_content(prompt)
            else:
                raise e
                
        return response.text
        
    except Exception as e:
        return f"⚠️ **Error generating AI strategy**: {str(e)}"
