
import streamlit as st
import requests
import chromadb
import ollama

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Website Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Website Knowledge Assistant")

st.markdown(
    "Ask questions about any website and get answers "
    "using Retrieval-Augmented Generation (RAG)."
)

st.divider()


# ==========================================
# 2. CONFIGURATION
# ==========================================

OLLAMA_MODEL = "qwen2.5-coder:3b"


# ==========================================
# 3. LOAD AI MODELS AND DATABASE
# ==========================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="website_knowledge",
        metadata={"hnsw:space": "cosine"}
    )

    return client, collection


embedding_model = load_embedding_model()
client, collection = load_chroma()


# ==========================================
# 4. EXTRACT WEBSITE CONTENT
# ==========================================

def extract_website_text(url):

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    for element in soup([
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "noscript"
    ]):
        element.decompose()

    main_content = soup.find("main")

    if main_content:
        text = main_content.get_text(
            separator=" ",
            strip=True
        )
    else:
        text = soup.get_text(
            separator=" ",
            strip=True
        )

    return text


# ==========================================
# 5. SPLIT TEXT INTO CHUNKS
# ==========================================

def split_into_chunks(
    text,
    chunk_size=500,
    overlap=100
):

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ==========================================
# 6. STORE DATA IN CHROMADB
# ==========================================

def store_website_content(url, chunks):

    existing = collection.get()

    if existing["ids"]:
        collection.delete(
            ids=existing["ids"]
        )

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    ).tolist()

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    metadata = [
        {
            "source": url,
            "chunk_number": i
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadata
    )


# ==========================================
# 7. RETRIEVE RELEVANT INFORMATION
# ==========================================

def retrieve_information(question):

    question_embedding = embedding_model.encode(
        [question]
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=min(3, collection.count())
    )

    if not results["documents"]:
        return []

    return results["documents"][0]


# ==========================================
# 8. GENERATE ANSWER USING OLLAMA
# ==========================================

def generate_answer(question, retrieved_chunks):

    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
    Answer the user's question using only the
    website information provided below.

    If the answer is not available in the website
    information, clearly say:

    "I could not find this information on the website."

    Do not invent facts.

    WEBSITE INFORMATION:
    {context}

    USER QUESTION:
    {question}
    """

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful website knowledge "
                    "assistant. Use only the supplied "
                    "website information to answer."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content


# ==========================================
# 9. WEBSITE INPUT SECTION
# ==========================================

st.header("🌐 1. Add a Website")

url = st.text_input(
    "Enter website URL",
    placeholder="https://www.python.org/"
)

if st.button(
    "🚀 Process Website",
    use_container_width=True
):

    if not url.startswith(
        ("http://", "https://")
    ):

        st.error(
            "Please enter a valid website URL."
        )

    else:

        try:

            with st.spinner(
                "Reading and processing website..."
            ):

                website_text = extract_website_text(
                    url
                )

                if not website_text.strip():
                    st.error(
                        "No readable content found."
                    )
                    st.stop()

                chunks = split_into_chunks(
                    website_text
                )

                if not chunks:
                    st.error(
                        "Could not create text chunks."
                    )
                    st.stop()

                store_website_content(
                    url,
                    chunks
                )

                st.session_state["website_url"] = url

                # Clear previous conversation
                st.session_state["messages"] = []

            st.success(
                "Website processed successfully!"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Words",
                len(website_text.split())
            )

            col2.metric(
                "Total Chunks",
                len(chunks)
            )

            col3.metric(
                "Stored Chunks",
                collection.count()
            )

        except Exception as e:

            st.error(
                f"Website processing failed: {e}"
            )


# ==========================================
# 10. CHAT INTERFACE
# ==========================================

st.divider()

st.header("💬 2. Chat with Your Website")

if "messages" not in st.session_state:
    st.session_state["messages"] = []


# Clear chat button

col1, col2 = st.columns([5, 1])

with col2:

    if st.button("🗑️ Clear Chat"):

        st.session_state["messages"] = []

        st.rerun()


# Display previous conversation

for message in st.session_state["messages"]:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# Chat input

question = st.chat_input(
    "Ask something about the website..."
)


if question:

    if collection.count() == 0:

        st.warning(
            "Please process a website first."
        )

    else:

        # Display user message

        st.session_state["messages"].append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)


        # Generate assistant answer

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching website and generating answer..."
            ):

                try:

                    retrieved_chunks = retrieve_information(
                        question
                    )

                    if not retrieved_chunks:

                        answer = (
                            "I could not find relevant "
                            "information on the website."
                        )

                    else:

                        answer = generate_answer(
                            question,
                            retrieved_chunks
                        )

                    st.markdown(answer)

                    st.session_state["messages"].append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:

                    st.error(
                        f"Could not generate answer: {e}"
                    )


# ==========================================
# 11. DATABASE INFORMATION
# ==========================================

st.divider()

st.header("📊 3. Database Information")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Stored Chunks",
    collection.count()
)

col2.metric(
    "AI Model",
    OLLAMA_MODEL
)

col3.metric(
    "Embedding Model",
    "all-MiniLM-L6-v2"
)

st.caption(
    "Powered by Streamlit, Ollama, "
    "Sentence Transformers and ChromaDB."
)
