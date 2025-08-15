# Overview

This repository implements a simple document-grounded chatbot built with LangChain, OpenAI models and a Gradio interface.

## Preparing the knowledge base
- `Process_files.py` scans the `Resources/` directory for PDFs, Word documents and plain text files.
- Each file is loaded and split into 1500 character chunks with slight overlap.
- The chunks are embedded with `OpenAIEmbeddings` and stored in a local Chroma vector database saved under `./data`.

## Running the chatbot
- `Chatbot.py` loads the persisted Chroma database and builds a `ConversationalRetrievalChain` around a GPT model.
- User queries are passed through this chain, which retrieves relevant chunks from the database and sends them, along with the conversation history, to the model.
- The response and the source pages for the retrieved documents are returned.
- A Gradio interface exposes this interaction, providing text boxes for questions, answers, and cited sources.

To try the bot, supply your OpenAI API key and run `Process_files.py` once to generate the database, then execute `Chatbot.py` to launch the Gradio UI.
