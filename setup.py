from setuptools import setup, find_packages

setup(
    name="src",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "src=src:run_app",
        ],
    },
    install_requires=[
        "streamlit>=1.40.2",
        "streamlit-float>=0.3.5",
        "langchain>=0.3.9",
        "langchain-openai>=0.2.9",
        "langchain-anthropic>=0.3.1",
        "langchain-community>=0.3.8",
        "langchain-core>=0.3.28",
        "langchain-text-splitters>=0.3.2",
        "langgraph>=0.2.53",
        "python-dotenv>=1.0.1",
        "pypdf>=5.1.0",
        "openai>=1.55.3",
        "anthropic>=0.42.0",
        "faiss-cpu",
    ],
)