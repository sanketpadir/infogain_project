#streamlit_app.py

# streamlit_app.py
import streamlit as st
import pandas as pd
from backend import load_data_into_sqlite, answer_query, TABLE_COLUMNS, TABLE_ORDER

st.set_page_config(page_title="Infogain Data Analysis Tool", layout="wide", page_icon="📊")

st.markdown("<h1 style='color:#1f3b73'>📊 Infogain Data Analysis Tool</h1>", unsafe_allow_html=True)
st.write("AI-driven SQL + Analytics demo — upload one or more CSVs and ask natural language questions.")

# Sidebar: upload or use sample data
st.sidebar.header("Data")
uploaded_files = st.sidebar.file_uploader("Upload CSV files (single or multiple)", type=["csv"], accept_multiple_files=True)
use_sample = st.sidebar.checkbox("Use bundled sample Data/dataset1.csv & Data/dataset2.csv", value=(len(uploaded_files) == 0))

# Load Data button
if "conn" not in st.session_state:
    st.session_state.conn = None
    st.session_state.loaded_tables = None

if st.sidebar.button("Load Data"):
    try:
        if uploaded_files and not use_sample:
            conn, schema = load_data_into_sqlite(uploaded_files)
        else:
            conn, schema = load_data_into_sqlite(None)
        st.session_state.conn = conn
        st.session_state.loaded_tables = schema
        st.success(f"Loaded tables: {', '.join(list(schema.keys()))}")
    except Exception as e:
        st.error(f"Error loading data: {e}")

# If not loaded, show instructions and stop gracefully
if st.session_state.conn is None:
    st.info("Please load data (upload CSV(s) or use bundled sample) from the sidebar to continue.")
    st.stop()

# Show detected schema
st.subheader("Loaded Tables & Columns")
for t, cols in (st.session_state.loaded_tables or {}).items():
    with st.expander(f"{t} ({len(cols)} columns)"):
        st.write(cols)

# Query input
st.subheader("Ask a question (natural language)")
question = st.text_area("Enter question:", placeholder="e.g., Which patients were diagnosed with thyroid disorders?")

temperature = st.slider("LLM temperature (0 deterministic → 1 creative)", 0.0, 1.0, 0.3, step=0.05)

if st.button("Generate Answer"):
    if not question.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Generating SQL & answer..."):
            try:
                result = answer_query(st.session_state.conn, question, temperature=temperature)

                st.subheader("Generated SQL")
                st.code(result["sql"])

                st.subheader(f"Rows Returned: {result['rows']}")
                st.dataframe(pd.DataFrame(result["preview"]))

                st.subheader("Data Summary")
                st.write(result["summary"])

                st.subheader("LLM Answer")
                st.info(result["answer"])

                st.subheader("Evaluation")
                st.json(result["evaluation"])

            except Exception as e:
                st.error(f"Error processing query: {e}")
