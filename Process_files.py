import sys
import os
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
import gradio as gr


os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY"

documents = []
for file in os.listdir("Resources"):
    if file.endswith(".pdf"):
        pdf_path = "./Resources/" + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        doc_path = "./Resources/" + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        text_path = "./Resources/" + file
        loader = TextLoader(text_path)
        documents.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data")
vectordb.persist()
