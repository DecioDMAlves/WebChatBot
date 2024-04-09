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

# Load the saved data
vectordb = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="./data")

pdf_qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo"),
    vectordb.as_retriever(search_kwargs={'k': 3}), max_tokens_limit=4000,
    return_source_documents=True,
    verbose=False
)

chat_history = []

def doc_bot_interface(query):
    global chat_history
    result = pdf_qa({"question": query, "chat_history": chat_history})

    answer = f"" + result["answer"]

    # Collecting information from all source documents
    page_info_list = []
    for doc in result['source_documents']:
        doc_name = list(doc)[1][1]['source'][12:-4]
        page_number = 1 + list(doc)[1][1]['page']
        page_info_list.append(f"página {page_number} do {doc_name}")

    # Join all page infos into a single string
    page_info = "Esta resposta foi baseada nas " + ", ".join(page_info_list) + "."

    return answer, page_info



# Set up Gradio interface
iface = gr.Interface(
    fn=doc_bot_interface,
    inputs=gr.Textbox(label="Question:", max_lines=3, autoscroll=False),
    outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Source")],
    live=False,
    flagging_options=[],
    title="ChatBot",
    description="This is a ChatBot based on XXXX"
)
iface.launch()
