#evaluation.py

def evaluate_response(question: str, answer: str, context: str):
    """
    Mock evaluation for offline usage without OpenAI.
    Returns a placeholder faithfulness score.
    
    This allows the app to run locally with Ollama Mistral
    without needing an OpenAI API key.
    """
    # Simple placeholder logic:
    # - If answer is non-empty, faithfulness = 1.0
    # - If answer is empty or clearly invalid, faithfulness = 0.0
    faithfulness_score = 1.0 if answer and answer.strip() else 0.0

    return {
        "faithfulness": faithfulness_score
    }
