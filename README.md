# Magic Conches


# Scholarly Peer Review App

**Magic Conches** is a Streamlit application designed to assist with the analysis and review of scientific papers. It leverages Large Language Models (LLMs) to provide a simplified, AI-powered simulation of the peer review process. The application allows users to upload research papers in PDF format, ask questions about the content, and generate a final feedback summary with a recommendation for acceptance or rejection.

## Features

*   **Multi-Modal Interaction:**
    *   **Simple Conversation Mode:** Engage in a basic conversation with an LLM before uploading a document.
    *   **Question & Answer Mode:**  Ask specific questions about an uploaded paper, either freeform or using pre-built questions designed to cover key aspects of a review.
    *   **Final Feedback Mode:** Generate a comprehensive review summary with a recommendation (accept/reject), and engage in follow-up discussion about the feedback.
*   **Document Processing:**
    *   Upload and process PDF research papers.
    *   Uses advanced retrieval techniques (Ensemble Retriever: BM25 + FAISS) to provide contextually relevant answers.
*   **Multiple LLM Support:**
    *   Utilizes both OpenAI and Anthropic models for diverse perspectives and enhanced answer quality.
*   **Dockerized Deployment:**
    *   Easily deployable using Docker and Docker Compose for consistent environments.
*   **Extensible Architecture:**
    *   Modular design with a clear separation of concerns (graph structure, state management, services, utilities).
    *   Ready for future expansion and feature additions.

## Project Structure

The project is organized with a clear and maintainable structure:

*   **`src/`:**  Contains all the application's Python code.
*   **`src/app.py`:** The main Streamlit application file.
*   **`src/graphs/`:** Contains subdirectories for each LangGraph graph (QA, Simple Conversation, Final Feedback). Each graph has its own `graph.py` and, optionally, `nodes/` and `state.py` files.
*   **`src/utils/`:**  Contains utility functions.
*   **`src/config.py`:** Centralized configuration for API keys, model names, and other settings.
*   **`.env_example`:**  An example file demonstrating the required environment variables.  *Never* commit your actual `.env` file with API keys.
*   **`Dockerfile` and `docker-compose.yml`:** Files for containerized deployment using Docker.

### Prerequisites

*   Python 3.9+
*   pip
*   Docker (optional, but recommended)
*   API keys for OpenAI and Anthropic (and Langchain, only for tracing)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://git.rwth-aachen.de/i5/teaching/bllma-lab/ws2024/magic-conches.git
    cd magic-conches
    ```

2.  **Create a virtual environment (highly recommended):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate     # On Windows (CMD)
    .venv\Scripts\Activate.ps1 # On Windows (PowerShell)
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file:**

    Copy the `.env_example` file to `.env`:

    ```bash
    cp .env_example .env
    ```

    Then, edit the `.env` file and add your *actual* API keys:

    ```
    OPENAI_API_KEY=your_openai_api_key
    ANTHROPIC_API_KEY=your_anthropic_api_key
    LANGCHAIN_API_KEY=your_langchain_api_key  # Required *only* if using LangSmith
    LANGCHAIN_TRACING_V2=false  # Set to "true" to enable LangChain tracing (optional)
    LANGCHAIN_PROJECT=your_langchain_project_name  # Optional, for LangSmith
    ```
    * **Important:** Replace `your_openai_api_key`, `your_anthropic_api_key`, and optionally `your_langchain_api_key` with your actual API keys.

### Running the Application

#### Option 1: Using Docker (Recommended)

This is the recommended way to run the application, as it ensures a consistent environment.

1.  **Build the Docker image:**

    ```bash
    docker compose build
    ```

2.  **Run the application:**

    ```bash
    docker compose up
    ```

    The application will be accessible at `http://localhost:8501`.

#### Option 2: Running Locally (without Docker)

1. **Activate the virtual environment** (if you have created):
    ```
     source .venv/bin/activate
    ```
2.  **Run the Streamlit application:**

    ```bash
    streamlit run src/app.py
    ```

    The application will be accessible at `http://localhost:8501` (or another port if 8501 is in use).

## Usage

1.  **Start in Conversation Mode:** You can begin by engaging in a general conversation with the LLM.
2.  **Upload a Paper:**  Use the file uploader in the sidebar to upload a research paper in PDF format.
3.  **Ask Questions:** Once a paper is uploaded, you can ask questions about it. You can type your own questions or use the pre-built questions to guide your analysis.
4.  **Generate Feedback:** Request a final feedback summary, including a recommendation for acceptance or rejection.
5.  **Follow-Up:** Ask follow-up questions about the generated feedback.

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with clear, descriptive commit messages.
4.  Push your branch to your fork.
5.  Submit a pull request to the `main` branch of the original repository.


## Acknowledgements
* This application uses the awesome Streamlit framework.
*  It utilizes the power of OpenAI and Anthropic's LLMs.
* Uses LangChain and LangGraph for the chat agent.

## Team Contributions

* **Davyd:**
    * **Group Coordinator**  – Managed team coordination and project planning.
    * **Main Architecture Design** – Defined and implemented the core structure and components of the application.
    * **User Interface Development** – Built an intuitive and user-friendly front end.
    * **LLM Integration** – Connected multiple LLMs (GPT-4o, GPT-4o-mini, Claude 3.5 Sonnet) to the application.
    * **Retrieval Development** – Implemented a retrieval mechanism to fetch relevant document information.
    * **Mid-Term and Final Presentation** – Prepared presentation for project milestones.
    * **Review Pull-Requests** – Conducted code reviews and provided feedback.
    * **Multi-Agent System Design** – Developed logic for managing multiple LLMs dynamically.
    * **Code Refactoring** -  Improved project structure, configuration, and maintainability.

* **Jannek:**
    * **Final Feedback** - Final usage mode for accept/reject recommendation and follow up questions.
    * **Application Structure Planning** - design of the application structure and modes.
    * **Prompt Engineering** - Engineered prompts in final feedback mode to handle follow up questions and conditional edges.
    * **Documentation** - README documentation and setup guide.
    * **Dockerization** - Dockerfile and docker-compose.yml for containerized deployment.
    * **Code Refactoring** - Improved project structure, configuration, and maintainability.
    * **Technical Report** - Formating and structuring of the final technical report.

* **Shohrukhbek:**
    * **Code Refactoring** -  Improved project structure, configuration, and maintainability; refactored app.py and feedback_graph.
    * **Pre-built Questions** - Implemented UI integration and logic for pre-built questions.
    * **Prompt Engineering** - Developed final feedback prompt, focusing on role, structure, and relevance.
    * **Dockerization** - Created Dockerfile and yml file for containerized deployment.
    * **Documentation** - Updated README.md with app features, setup, usage and contributions.
    * **Architectural Participation** - Contributed to architectural discussions on code structure and maintainability.
    * **Mid-Term Presentation** - Prepared the mid-term project presentation.

* **Tsan-Yu:**
    * **Paper Finding** - Found a relevant research that supports the idea of LLMs as helpful peer reviewers.
    * **Memory Development** - Implemented the component where the chat history is summarized and the length of messages is controlled.
    * **Memory Integration** - Connected the memory component to the application.
    * **Prompt Engineering** - Made the question-answering feedback more critical instead of always positive/cheerful.
    * **Application Structure Planning** - design of the application structure and modes.
    * **Code Refactoring** - Improved project structure, configuration, and maintainability.
    * **Technical Report** - Formating and structuring of the final technical report.

