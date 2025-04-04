import os
import subprocess 
from dotenv import load_dotenv
from flask import send_from_directory
from src.prompt import SYSTEM_PROMPT
from flask import Flask, render_template, request
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.documents import Document

app = Flask(__name__)

load_dotenv()

CHROMA_DB_PATH = "chroma_storage"
CACHE_DB_PATH = "cache_storage"

embeddings = AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_EMBED_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )


if os.path.isdir(CHROMA_DB_PATH):
    docsearch = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    print("Chroma database found.")
else:
    subprocess.run(["python", "store_index.py"])
    docsearch = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    print("Chroma database not found. Creating a new one.")

if os.path.isdir(CACHE_DB_PATH):
    cache_store = Chroma(persist_directory=CACHE_DB_PATH, embedding_function=embeddings)
else:
    os.makedirs(CACHE_DB_PATH, exist_ok=True)
    cache_store = Chroma(persist_directory=CACHE_DB_PATH, embedding_function=embeddings)

    
llm = AzureChatOpenAI(
    model=os.getenv("DEPLOYMENT_NAME"),
    api_key=os.environ["AZURE_OPENAI_API_KEY"] ,  
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.environ["ENDPOINT_URL"]
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}\n\nContext:\n{context}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)

def rag_chain_with_links(question):

    question_embedding = embeddings.embed_query(question)
    cached = cache_store._collection.query(
        query_embeddings=[question_embedding],
        n_results=1,
        include=["documents", "distances", "metadatas"]
    )


    if (cached["documents"] and cached["documents"][0]
        and cached["distances"] and cached["distances"][0]
        and cached["distances"][0][0] < 0.3):

        cached_answer = cached["documents"][0][0]
        cached_meta = cached["metadatas"][0][0]
        source = cached_meta.get("source", "#")
        page = cached_meta.get("page", 1)
        print("🔁 Cache hit!")

        return {
            "answer":
            f"""
                {cached_answer}<br><br>
                <a href="Data/{source}#page={page}" target="_blank">
                    🔗 Open PDF (Page {page})
                </a><br><br>
                <i style='color:gray;'>⚡ Answer from Cache</i>
            """
        }

    print("🔍 Cache miss. Calling LLM...")

    results = docsearch.similarity_search_with_score(question, k=3)
    context = "\n\n".join([doc.page_content for doc, _ in results])


    top_doc = results[0][0] if results else Document(page_content="", metadata={})
    response = llm.invoke(prompt.format(input=question, context=context))


    cache_store.add_documents([
    Document(
        page_content=response.content,
        metadata={
            "question": question,
            "source": top_doc.metadata.get("source", "Unknown"),
            "page": top_doc.metadata.get("page", 1)
            }
        )
    ])
    print("✅ Answer added to cache.")

    sources = "".join([
        f"""
        <details style='margin-bottom: 10px;'>
            <summary>
                Source {i+1} – {doc.metadata.get("source", "Unknown")} / Page {doc.metadata.get("page", 1)}
            </summary>
            <a href="Data/{doc.metadata.get("source", "#")}#page={doc.metadata.get("page", 1)}" target="_blank">
                🔗 Open PDF (Page {doc.metadata.get("page", 1)})
            </a><br><br>
            <pre style="white-space: pre-wrap;">{doc.page_content}</pre>
        </details>
        """
        for i, (doc, _) in enumerate(results)
    ])

    return {"answer": f"{response.content}<br><br>{sources}"}


@app.route("/")
def index():
    return render_template('chat.html')

@app.route('/Data/<path:filename>')
def serve_pdf(filename):
    return send_from_directory('Data', filename)

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("User: ", msg)
    response = rag_chain_with_links(msg)
    return response["answer"]

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)