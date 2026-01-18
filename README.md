# Infogain Health Data Analysis Tool – GenAI & RAG

A **Streamlit-based GenAI tool** for health data analytics that allows users to ask **natural language questions** on multiple CSV datasets. The system automatically generates SQL queries, executes them on SQLite, summarizes results, and produces human-readable answers using **Ollama Mistral LLM**. It also evaluates answer quality using **RAGAS metrics**.

---

## Features

- Natural Language SQL Generation using **Ollama Mistral**.
- Safe SQL execution with **sanitization** and `LIMIT 200`.
- Handles **joins** across multiple datasets (`Dataset1` and `Dataset2`).
- Supports **multi-disorder queries** and **aggregates**.
- Summarizes query results with row count, sample data, and statistics.
- Generates **human-readable answers** from SQL results.
- Evaluates responses using **RAGAS metrics**: faithfulness, context precision, context recall.
- Thread-safe SQLite in-memory database for fast query execution.

---

## Project Structure

infogain_project/
│
├─ app/
│ ├─ streamlit_app.py # Streamlit UI
│ ├─ backend.py # Core logic for SQL generation, execution, summarization, and LLM answers
│ ├─ prompts.py # LLM prompt templates
│ ├─ evaluation.py # RAGAS evaluation logic
│
├─ Data/
│ ├─ dataset1.csv
│ ├─ dataset2.csv
│
├─ README.md


---

## Setup Instructions

1. **Clone the repository:**

```bash
git clone <repository-url>
cd infogain_project

2. **Create a virtual environment (optional but recommended):**

python -m venv venv
venv\\Scripts\\activate     # Windows

3. Install required packages:

pip install -r requirements.txt
pip install python-pptx      # for presentation generation

4. Install and run Ollama

ollama pull mistral


Running the Application

1. Start the Streamlit app:

streamlit run app/streamlit_app.py

2. Load Data: Click Load Data to load dataset1.csv and dataset2.csv into SQLite.

3. Ask Questions: Enter natural language questions in the text area. Examples:
1. give me Patient_Number which has Sex is 1 and Chronic_kidney_disease is 0 from dataset1?
2. which Patient_Number has maximum Age and BMI from dataset1?
3. count all rows from dataset2?

4. View Output: The app displays:
Generated SQL
Rows returned (preview)
Data summary
Final answer from LLM
RAGAS evaluation metrics