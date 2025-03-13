from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from src.config import (
    OPENAI_MODEL_GPT4O,
    OPENAI_MODEL_GPT4O_MINI,
    ANTHROPIC_CLAUDE_3_5_SONNET,
    OPENAI_API_KEY, 
    ANTHROPIC_API_KEY
)


def get_openai_llm(model_name = OPENAI_MODEL_GPT4O): 
    """Returns an initialized ChatOpenAI instance."""
    return ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY)

def get_anthropic_llm(model_name = ANTHROPIC_CLAUDE_3_5_SONNET):
    """Returns an initialized ChatAnthropic instance."""
    return ChatAnthropic(model=model_name, api_key=ANTHROPIC_API_KEY)