import streamlit as st
import os
import hashlib
import random
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

load_dotenv()
embeddings = OpenAIEmbeddings()

st.set_page_config(page_title="📘 RAG PDF Assistant", page_icon="🤖", layout="wide")
st.title("📘 RAG PDF Assistant")
st.write("Upload a PDF and ask questions interactively!")

# Sidebar instructions
with st.sidebar:
    st.header("ℹ️ About this app")
    st.markdown(
        """
        This is a **RAG (Retrieval-Augmented Generation) app** built with LangChain.  
        It allows you to:
        1. **Upload any PDF** (book, research paper, notes, etc.).  
        2. The app will **index** the PDF using embeddings + FAISS.  
        3. Ask **questions** in natural language.  
        4. Get answers **grounded in your document**.  

        ⚡ Powered by OpenAI + LangChain  
        """
    )

# Generate hash for caching vectorstore
def file_hash(file):
    return hashlib.md5(file.read()).hexdigest()

# Vectorstore builder
@st.cache_resource
def build_vectorstore(pdf_path, cache_key):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(f"faiss_{cache_key}")
    return vectorstore

uploaded_file = st.file_uploader("📂 Upload your PDF", type=["pdf"])

funny_loaders = [
    "📂 Uploading your file... Looks heavy, might take up to 59 seconds 😉",
    "📖 Reading your PDF... AI needs glasses for this one 🤓",
    "⚡ Crunching text faster than your boss reads reports...",
    "🐐 Feeding your PDF to AI goats... they chew slow sometimes",
    "🧘‍♂️ Counting words like a monk before enlightenment..."
]

if uploaded_file is not None:
    pdf_path = "uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    cache_key = file_hash(uploaded_file)

    # Fun sarcastic spinner
    with st.spinner(random.choice(funny_loaders)):
        if os.path.exists(f"faiss_{cache_key}"):
            vector_store = FAISS.load_local(
                f"faiss_{cache_key}", embeddings, allow_dangerous_deserialization=True
            )
        else:
            vector_store = build_vectorstore(pdf_path, cache_key)

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(temperature=0)

    prompt = PromptTemplate(
        template=(
            "You are a PDF assistant. Use ONLY the following context to answer.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}"
        ),
        input_variables=["context", "question"]
    )

    parser = StrOutputParser()
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    st.subheader("💬 Ask questions about your PDF")

    query = st.text_input("Type your question here:")

    if query:
        result = chain.invoke(query)
        st.session_state.history.append(("🧑 You", query))
        st.session_state.history.append(("🤖 Assistant", result))

    for speaker, msg in st.session_state.history:
        st.markdown(f"**{speaker}:** {msg}")
