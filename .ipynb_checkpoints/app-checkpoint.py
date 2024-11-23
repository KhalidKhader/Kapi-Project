import streamlit as st
import joblib
import httpx
from string import Template
import pandas as pd
from sklearn.preprocessing import LabelEncoder


OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_CONFIG = {
    "model": "mistral:7b-instruct-v0.2-q4_K_S",
    "keep_alive": "5m",
    "stream": False,
}

PROMPT_TEMPLATE = Template(
    """Extract the following details from the job description below, Ensure each response is concise, accurate, and formatted as specified:
    - **Years of Experience**: Extract the required experience as a single number representing years. If it's less than 1 year, use "0" (without text).
    - **Education Level**: Extract the education level choose one(choose from: School, High School, Diplomate, College, Bachelor, Master, PhD, Other).
    - **Job Level**: Extract the level of the role (choose from: Senior, Mid, Other, Junior, Manager, Executive, Assistant, Intern, Lead, Director).
    - **Job Sector**: Provide the job sector (choose one: Technology, Healthcare, Education, Retail, Manufacturing, Other).
    - **Gender**: Extract gender (choose one: Male, Female, Other).
    - **Age**: Extract age (choose a number from 18 to 100, or Other if not available).

    Job Description: $text

    Respond in JSON format with the keys: "Job Title", "Years of Experience", "Expected Salary Range", "Expected Salary Currency", "Job Type", "Location", "Education Level", "Major", "Job Level", "Job Function", "Job Sector", "Gender", "Age".
    As 
    - "Years of Experience"
    - "Education Level"
    - "Job Level"
    - "Job Sector"
    - "Gender"
    - "Age"
    For missing fields, use the value "Other".
    """
)

PROMPT_TEMPLATE_OLLAMA = Template(
    """EXPECT the Salary for this person $text 
    """
)

def extract_job_details(description):
    """Call Llama model to extract job details."""
    prompt = PROMPT_TEMPLATE.substitute(text=description)
    response = httpx.post(
        OLLAMA_ENDPOINT,
        json={"prompt": prompt, **OLLAMA_CONFIG},
        headers={"Content-Type": "application/json"},
        timeout=240,
    )
    if response.status_code != 200:
        st.error(f"Error {response.status_code}: {response.text}")
        return None

    try:
        result = response.json()["response"].strip()
        return eval(result)  
    except Exception as e:
        st.error(f"Error parsing response: {e}")
        return {
            "Years of Experience": 0,
            "Education Level": "Other",
            "Job Level": "Other",
            "Job Sector": "Other",
            "Gender": "Other",
            "Age": 25,
        }

def extract_salary(description):
    """Call Llama model to extract job details."""
    prompt = PROMPT_TEMPLATE_OLLAMA.substitute(text=description)
    response = httpx.post(
        OLLAMA_ENDPOINT,
        json={"prompt": prompt, **OLLAMA_CONFIG},
        headers={"Content-Type": "application/json"},
        timeout=240,
    )
    if response.status_code != 200:
        st.error(f"Error {response.status_code}: {response.text}")
        return None

    try:
        result = response.json()["response"].strip()
        return result
    except Exception as e:
        st.error(f"Error parsing response: {e}")



def preprocess_features(features):
    """Map extracted features to the required categories and handle missing values."""
    # Define category mappings
    education_mapping = {
        "School": "Other",
        "High School": "High School",
        "Diplomate": "Diploma",
        "College": "Bachelor's Degree",
        "Bachelor": "Bachelor's Degree",
        "Master": "Master's Degree",
        "PhD": "PhD",
    }
    job_level_mapping = {
        "Senior": "Senior",
        "Mid": "Junior",  
        "Junior": "Junior",
        "Manager": "Manager",
        "Executive": "Executive",
        "Assistant": "Assistant",
        "Intern": "Intern",
        "Lead": "Lead",
        "Director": "Director",
        "Other": "Other",
    }
    job_sector_mapping = {
        "Technology": "Engineering & Technology",
        "software": "Engineering & Technology", 
        "developer": "Engineering & Technology",
        "it": "Engineering & Technology",
        "data": "Engineering & Technology", 
        "scientist": "Engineering & Technology",
        "Healthcare": "Healthcare",
        "Education": "Education",
        "Retail": "Retail",
        "Manufacturing": "Manufacturing",
        "Other": "Other",
    }

    # Gender mapping
    gender_mapping = {
        "Male": "Male",
        "Female": "Female",
        "Other": "Other",
    }

    # Preprocess features
    features["Education Level"] = education_mapping.get(features.get("Education Level", "Other"), "Other")
    features["Job Level"] = job_level_mapping.get(features.get("Job Level", "Other"), "Other")
    features["Job Sector"] = job_sector_mapping.get(features.get("Job Sector", "Other"), "Other")
    features["Gender"] = gender_mapping.get(features.get("Gender", "Other"), "Other")

    # Handle missing "Years of Experience" and convert to numeric
    features["Years of Experience"] = float(features.get("Years of Experience", 2))  
    features["Age"] = 25 if features.get("Age", 25) == 'Other' else int(features.get("Age", 25))

    # Handle missing Age (assuming age is given as a number or "Other")
    features["Age"] = features.get("Age", "Other")
    
    st.write(features)

    return features


@st.cache_resource
def load_model():
    return joblib.load("rf_pipe_model.pkl")

model = load_model()

def main():
    st.title("Job Description Analysis and Prediction")
    
    job_description = st.text_area("Enter the Job Description:")
    
    if st.button("Analyze and Predict"):
        if job_description.strip():
            extracted_features = extract_job_details(job_description)
            
            if extracted_features:
                processed_features = preprocess_features(extracted_features)

                input_data = pd.DataFrame([processed_features])
                data = input_data[["Age", "Gender", "Education Level", "Job Sector","Job Level", "Years of Experience"]]
                
                prediction = model.predict(data)

                st.write("**Extracted Details:**", extracted_features)
                st.write("**Preprocessed Features for Model:**", processed_features)
                st.success(f"Predicted Job Level: {prediction[0]}")
        else:
            st.error("Please provide a valid job description.")

    if st.button("Predict Salary from Ollama Model"):
        if job_description.strip():
            salary_text = extract_salary(job_description)
            if salary_text:
                st.success(salary_text)
            else:
                st.error("Please provide a valid job description.")
        else:
            st.error("Please provide a valid job description for salary prediction.")

if __name__ == "__main__":
    main()
