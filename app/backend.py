# backend.py

import sqlite3  # For creating an in-memory SQLite database to query CSVs like SQL tables
import pandas as pd
import re # For regex operations, mainly for safe column qualification and table name sanitization.
from typing import List, Tuple, Dict, Optional # Adds type hints for clarity.
from prompts import SQL_GEN_PROMPT, ANSWER_GEN_PROMPT
from evaluation import evaluate_response
from langchain_ollama import ChatOllama
from langchain.chains import LLMChain

# Global schema storage 
TABLE_COLUMNS: Dict[str, List[str]] = {} # Track columns and table order for all loaded datasets, used in dynamic schema handling.
TABLE_ORDER: List[str] = [] 


# LOAD CSV(S) INTO SQLITE (Thread-safe)

def load_data_into_sqlite(uploaded_files: Optional[List] = None) -> Tuple[sqlite3.Connection, Dict[str, List[str]]]:
    """
    Load CSVs into an in-memory SQLite DB. If uploaded_files provided (list of file-like objects),
    each file is loaded as a table named from the filename (sanitized). Otherwise falls back to
    Data/dataset1.csv and Data/dataset2.csv (legacy behavior).
    """
    global TABLE_COLUMNS, TABLE_ORDER
    TABLE_COLUMNS = {}
    TABLE_ORDER = []

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    
# Table name sanitization
# 1. Converts filenames to valid table names.
# 2. Replaces invalid characters, removes .csv, and adds prefix if name starts with a digit.

    def _table_name_from_filename(name: str) -> str:
        
        base = re.sub(r"[^0-9a-zA-Z_]", "_", name.split("/")[-1].split("\\")[-1])
        base = re.sub(r"\.csv$", "", base, flags=re.IGNORECASE)
        
        if re.match(r"^\d", base):
            base = "t_" + base
        return base or "table"
    
# Loading files
# 1. Loads each CSV into SQLite as a table.
# 2. Stores columns and order for schema awareness.
# 3. Supports both dynamic CSV upload and default datasets.

    if uploaded_files:
        for f in uploaded_files:
            
            filename = getattr(f, "name", "uploaded.csv")
            tbl = _table_name_from_filename(filename)
            
            df = pd.read_csv(f)
            df.to_sql(tbl, conn, index=False, if_exists="replace")
            cols = df.columns.tolist()
            TABLE_COLUMNS[tbl] = cols
            TABLE_ORDER.append(tbl)
    else:

        try:
            df1 = pd.read_csv("Data/dataset1.csv")
            df2 = pd.read_csv("Data/dataset2.csv")
        except FileNotFoundError as e:
            raise FileNotFoundError("No uploaded files and fallback Data CSVs not found.") from e

        df1.to_sql("Dataset1", conn, index=False, if_exists="replace")
        df2.to_sql("Dataset2", conn, index=False, if_exists="replace")
        TABLE_COLUMNS["Dataset1"] = df1.columns.tolist()
        TABLE_COLUMNS["Dataset2"] = df2.columns.tolist()
        TABLE_ORDER = ["Dataset1", "Dataset2"]

    return conn, TABLE_COLUMNS


# Build schema block for prompt injection
# 1. Returns a human-readable schema for the LLM prompt.

def get_schema_block() -> str:
    """
    Returns a human-readable schema block communicating tables and columns for the LLM.
    Example:
      Tables:
      - patients (patient_id, age, gender)
      - records (record_id, patient_id, price)
    """
    lines = ["Tables and columns available:"]
    for tbl in TABLE_ORDER:
        cols = TABLE_COLUMNS.get(tbl, [])
        lines.append(f"- {tbl}({', '.join(cols)})")
    return "\n".join(lines)


# SQL Sanitization
# 1. Only the first SQL statement runs (prevents injection).
# 2. Adds a LIMIT 200 to prevent huge outputs.
# 3. If the SQL is invalid, returns a safe fallback query.

def sanitize_sql(sql: str) -> str:
    first_sql = sql.split(";")[0].strip()
    allowed_keywords = ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE")
    if not first_sql.upper().startswith(allowed_keywords):
        return "SELECT 1 LIMIT 1"
    if "LIMIT" not in first_sql.upper():
        first_sql += " LIMIT 200"
    return first_sql


# Safe Column Qualification (dynamic)
# 1. Ensures all columns are fully qualified with table names (table.column).
# 2. Handles ambiguous column names across multiple tables.
# 3. Uses regex to avoid modifying string literals in queries.
# 4. Keeps a preference order from TABLE_ORDER.

def qualify_columns(sql: str) -> str:
    """
    Safely qualify column names using detected table schemas.
    If a column appears in multiple tables, we disambiguate using TABLE_ORDER preference.
    """
    if not TABLE_COLUMNS:
        return sql


    col_to_tables: Dict[str, List[str]] = {}
    for tbl in TABLE_ORDER:
        for c in TABLE_COLUMNS.get(tbl, []):
            col_to_tables.setdefault(c, []).append(tbl)

    
    def replace_in_segment(segment: str) -> str:
        
        for col, tables in col_to_tables.items():
            
            pref_table = tables[0] 
            
            pattern = rf"(?<!\.)\b{re.escape(col)}\b"
            replacement = f"{pref_table}.{col}"
            segment = re.sub(pattern, replacement, segment)
        return segment

    
    parts = re.findall(r"(?:'[^']*'|\"[^\"]*\"|[^'\";]+)", sql)
    new_parts = [replace_in_segment(p) if not (p.startswith("'") or p.startswith('"')) else p for p in parts]
    return "".join(new_parts)


# RUN SQL
# 1. Executes the sanitized SQL query on SQLite.
# 2. Returns a pandas dataframe.

def run_sql(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    safe_sql = sanitize_sql(sql)
    return pd.read_sql_query(safe_sql, conn)


# SUMMARIZE DATAFRAME (sampled to cap heavy summarization)
# 1. Returns:Number of rows,Sample of top rows,Summary statistics

def summarize_df(df: pd.DataFrame, max_sample: int = 500) -> Tuple[str, dict]:
    n = len(df)
    sample_df = df.head(max_sample) if n > max_sample else df
    head = sample_df.head(10).to_dict(orient="records")
    stats = sample_df.describe(include="all").to_dict()
    summary = f"Rows Returned: {n}\nSample Rows: {head}\nStatistics (sampled up to {max_sample} rows): {stats}"
    return summary, {"n": n}


# SQL GENERATION USING OLLAMA MISTRAL (schema-aware)
# 1. Uses LLM (Mistral) to generate SQL dynamically.
# 2. Injects schema block so LLM knows table/column names.

def generate_sql_mistral(question: str, temperature: float = 0.0) -> str:
    """
    Generate SQL while injecting the dynamic schema block into the prompt.
    """
    schema_block = get_schema_block()
    llm = ChatOllama(model="mistral", temperature=temperature)
    chain = LLMChain(llm=llm, prompt=SQL_GEN_PROMPT)
    # run with both variables expected by the prompt template
    sql_msg = chain.run({"user_question": question, "schema_block": schema_block})
    return sql_msg.strip()


# ANSWER GENERATION USING OLLAMA MISTRAL
# 1. Uses LLM to explain results in natural language.
# 2. Uses summary of query results as context.

def generate_answer_mistral(question: str, summary: str, temperature: float = 0.3) -> str:
    llm = ChatOllama(model="mistral", temperature=temperature)
    chain = LLMChain(llm=llm, prompt=ANSWER_GEN_PROMPT)
    answer_msg = chain.run({"user_question": question, "data_summary": summary})
    return answer_msg.strip()


# FULL PIPELINE

def answer_query(conn: sqlite3.Connection, user_question: str, temperature: float = 0.3):
    """
    Top-level: generate SQL using schema, qualify columns, execute, summarize, answer, evaluate.
    """
    # 1. Generate SQL using schema-aware prompt
    sql = generate_sql_mistral(user_question, temperature=temperature)

    # 2. Qualify columns safely using detected schema
    sql = qualify_columns(sql)

    # 3. Execute SQL
    df = run_sql(conn, sql)

    # 4. Summarize (sampled)
    summary, meta = summarize_df(df)

    # 5. Generate answer using summary
    answer = generate_answer_mistral(user_question, summary, temperature=temperature)

    # 6. Evaluate
    evaluation = evaluate_response(user_question, answer, summary)

    return {
        "sql": sql,
        "rows": meta["n"],
        "preview": df.head(10).to_dict(orient="records"),
        "summary": summary,
        "answer": answer,
        "evaluation": evaluation,
    }
