# RAG + CAG System Summary
Hybrid Retrieval with Caching
This app combines Retrieval-Augmented Generation (RAG) with Cache-Augmented Generation (CAG). It first checks for previously answered similar questions from a local vector cache.

Local Vector Store via Chroma
Documents and previous answers are embedded using Azure OpenAI and stored in Chroma DB. A separate cache store is used for fast similarity-based retrieval of prior responses.

Semantic Caching
If a new query is at least 80% semantically similar (distance < 0.2) to a cached one, the system returns the previous answer instantly — including a link to the original PDF page.

Fallback to RAG Pipeline
If no suitable cached response is found, the system performs a fresh RAG: it retrieves relevant chunks and generates an answer using Azure GPT models.

Source-Linked Responses
All answers (cached or new) include collapsible references with links to the exact PDF page the answer was retrieved from.

Flask-Powered UI
A simple Flask app handles chat communication and PDF linking through a clean, user-friendly interface.

# How to run?
### STEPS:

### STEP 01- Create a .venv environment 

```bash
python -m venv .venv
```

```bash
.\.venv\Scripts\activate.ps1
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your openai credentials as follows:

```ini
AZURE_OPENAI_DEPLOYMENT_NAME="text-embedding-ada-002"
AZURE_OPENAI_EMBED_API_VERSION="2023-05-15"
AZURE_OPENAI_ENDPOINT="xxxxxxxxxx"
AZURE_OPENAI_API_KEY="xxxxxxxxxxx"

DEPLOYMENT_NAME="gpt-4o"
ENDPOINT_URL="xxxxxxx"
AZURE_OPENAI_API_VERSION="2024-05-01-preview"
```


```bash
# run the following command to store embeddings
python store_index.py
```

```bash
# Finally run the following command
python app.py
```


### Techstack Used:

- Python
- LangChain
- Flask
- Azure OpenAI
- ChromaDB
