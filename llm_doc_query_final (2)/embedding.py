from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings

def load_document(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)

def create_vectorstore(chunks):
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")
    return FAISS.from_documents(chunks, embeddings)