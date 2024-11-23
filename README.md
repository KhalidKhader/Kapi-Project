# Job Description Analysis and Salary Prediction App

This project is a web application that analyzes a given job description and predicts job level and salary based on extracted features. The app integrates a machine learning model for job level prediction and a request to the Ollama model for salary prediction based on the job description.

## Project Structure

```
.
├── Extract data from text using llama.ipynb            
├── Regression Task for Salaries.ipynb
├── Word_Embedeeing,_using_RNN_LSTM_and_BERT.ipynb
├── data processing.ipynb
├── data ├── LLM_Expected_salary.csv
         ├── Salary_Data.csv
         ├── job_descriptions_for_ml.csv
         ├── job_descriptions_modified.csv
         ├── job_descriptions_with_extracted_columns1.csv
         ├── positions_salaries.csv
         ├──processed_job_descriptions_for_ml.csv
├── PDF and docs├── Solution doc.pdf
                ├── Data Scientist Assignment.pdf
├── GUI using Streamlit ├── scrap.py
                        ├── app.py
                        ├── rf_pipe_model.pkl
├── requirements.txt
└── README.md           
```

## Technologies Used

- **Streamlit**: A fast and easy-to-use framework for building interactive web applications.
- **Scikit-learn**: For the machine learning pipeline to process input data and make predictions.
- **Ollama API**: Used for extracting salary predictions from the job description text.
- **Pandas**: For data manipulation and handling of input features.
- **Httpx**: For making API requests to the Ollama service.

## Installation

To run the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/job-description-analysis.git
   cd job-description-analysis
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**:
   ```bash
   cd "GUI using Streamlit"
   streamlit run app.py
   ```

## Project Overview

This application has two key features:

### 1. **Job Level Prediction**
   The app analyzes a job description and predicts the **job level** based on extracted features like:

   - Age
   - Gender
   - Education Level
   - Job Title
   - Years of Experience

   The features are preprocessed and passed into a pre-trained machine learning model that outputs the predicted job level.

### 2. **Salary Prediction**
   Using the job description, the app makes a **salary prediction** using the Ollama API. The salary is extracted directly from the model's response and cleaned before being returned to the user.

### Components Breakdown

#### **1. `app.py`** (Main Application Script)

This is the main entry point of the application and is built using the Streamlit framework. The app has the following sections:

- **Job Description Input**: Users can enter a job description to analyze.
- **Buttons for Predictions**: There are two buttons:
  - **Analyze and Predict Job Level**: This button triggers the job level prediction.
  - **Predict Salary from Ollama Model**: This button makes a request to the Ollama API to predict salary based on the job description.

#### **2. `extract_job_details` Function**
   This function is responsible for extracting the key features from the job description, such as:

   - Age
   - Gender
   - Education Level
   - Job Title
   - Years of Experience
   
   The extracted details are then preprocessed for input into the machine learning model.

#### **3. `preprocess_features` Function**
   This function handles the preprocessing of the extracted features before they are passed to the model. This step may involve scaling, encoding categorical features, and handling missing values.

#### **4. `extract_salary` Function**
   This function makes a request to the Ollama API to extract the salary prediction based on the job description. The response is cleaned and formatted into a usable salary value.

#### **5. Machine Learning Model (Job Level Prediction)**
   The job level prediction is made using a pre-trained machine learning model. The model is designed to output the job level based on the provided features.

### Flow of the Application

1. **User Input**: The user enters a job description in a text box.
2. **Feature Extraction**: The app extracts key features (age, gender, etc.) from the job description using natural language processing techniques.
3. **Preprocessing**: These extracted features are preprocessed (e.g., handling missing values, scaling numerical features, encoding categorical features).
4. **Job Level Prediction**: The preprocessed features are passed into a trained model to predict the job level.
5. **Salary Prediction**: The app sends the job description to the Ollama API for salary prediction and displays the result.
6. **Output**: The app shows the extracted features and the predicted job level and salary.

## API Integration: Ollama API

The Ollama API is used to predict salaries based on the job description. The `extract_salary` function makes an HTTP POST request to the Ollama endpoint with the job description, processes the response, and returns the predicted salary.

### Example of a Salary Prediction Request:

```python
response = httpx.post(
    OLLAMA_ENDPOINT,
    json={"prompt": prompt, **OLLAMA_CONFIG},
    headers={"Content-Type": "application/json"},
    timeout=240,
)
```

#### Important Notes:
- Ensure that the **Ollama API endpoint** (`OLLAMA_ENDPOINT`) is configured properly.
- The salary prediction might include unexpected symbols (e.g., `€` or `$`). The app removes these symbols and attempts to extract a numeric salary value.

## Dependencies

The following Python libraries are required to run this app:

- `streamlit` – For the interactive web interface.
- `scikit-learn` – For the machine learning model and preprocessing.
- `pandas` – For data handling and manipulation.
- `httpx` – For making HTTP requests to the Ollama API.
- `template` – For creating dynamic prompts for the Ollama model.

You can install all the dependencies by running:

```bash
pip install -r requirements.txt
```

## Requirements File (`requirements.txt`)

```
streamlit
scikit-learn
pandas
httpx
```

## Troubleshooting

- **Invalid Response from Ollama API**: If the salary response contains unexpected characters, you can enhance the response parsing logic to handle different formats or additional symbols.
  
- **Missing Features or Incorrect Predictions**: If the extracted features seem incomplete or incorrect, you may need to refine the feature extraction function to handle different formats or edge cases in job descriptions.

## Contributing

Contributions to the project are welcome! Feel free to fork the repository, submit issues, and create pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Key Points:
- The **README** explains the purpose of the project, how to set it up, the overall workflow, and details about each component.
- It also includes instructions on installing dependencies, running the app, and troubleshooting common issues.

Let me know if you'd like any further adjustments to this!
