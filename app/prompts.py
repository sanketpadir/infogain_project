#prompts.py

# prompts.py
from langchain.prompts import PromptTemplate

SQL_GEN_PROMPT = PromptTemplate(
    template="""
You are an expert SQL assistant. Below are the tables and columns you can use:

{schema_block}

User question:
\"\"\"{user_question}\"\"\"

Rules:
- Return ONLY valid SQLite SQL.
- Use only the tables and columns provided above.
- If a join is needed, use the appropriate column(s) from the schema (e.g., patient id).
- Add LIMIT 200 at the end of the first statement.
- Do NOT write explanations; return only the SQL statement.

SQL:
""",
    input_variables=["user_question", "schema_block"]
)

ANSWER_GEN_PROMPT = PromptTemplate(
    template="""
You are a helpful health-data analysis assistant.

User Question:
{user_question}

Data Summary:
{data_summary}

Instructions:
- Provide a clear 3–6 sentence explanation.
- Mention the number of rows (N).
- Use only the provided data.
- This is NOT medical advice.
- If data is insufficient, clearly state it.

Answer:
""",
    input_variables=["user_question", "data_summary"]
)
