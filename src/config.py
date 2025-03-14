import os
from dotenv import load_dotenv

load_dotenv()

# LLM models
OPENAI_MODEL_GPT4O = "gpt-4o"
OPENAI_MODEL_GPT4O_MINI = "gpt-4o-mini" 
OPENAI_MODEL_EMBEDDING = "text-embedding-3-large"
ANTHROPIC_CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022" 

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Retriever related parameters
BM25_K = 2
FAISS_K = 2
ENSEMBLE_WEIGHTS = [0.5, 0.5]