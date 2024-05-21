# import streamlit as st
# from PyPDF2 import PdfReader
# from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.llms import OpenAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.callbacks import get_openai_callback
# import pickle
# import os
# from langchain.chat_models import ChatOpenAI
#
# st.header('ChatPDF v0.1')
# st.sidebar.header(":blue[Welcome to ChatPDF!]")
# pdf = st.file_uploader('Upload a PDF file with text in English. PDFs that only contain images will not be recognized.', type=['pdf'])
# query = st.text_input('Ask question about the PDF you entered!', max_chars=300)
# try:
#     pdf_doc = PdfReader(pdf)
#     for page in pdf_doc.pages:
#         txt += page.extract_text()
#
# except Exception as e:
#     st.error(str(e))
# text_split = RecursiveCharacterTextSplitter(
#             chunk_size=1000, # number of characters per chunk
#             chunk_overlap=200, # used to keep the context of a chunk intact with previous and next chunks
#             length_function=len
#         )
#         chunks = text_split.split_text(text=txt)
# embeddings = OpenAIEmbeddings()
# vectorStore = FAISS.from_texts(chunks,embedding=embeddings)
# docs = vectorStore.similarity_search(query=query)
# llm = ChatOpenAI(model_name="gpt-3.5-turbo")
# chain = load_qa_chain(llm=llm, chain_type="stuff")
# response = chain.run(input_documents=docs, question=query)
# st.write(response)
# with open(f"{store_name}.pkl", "rb") as f:
#     vs = pickle.load(f)
# with open(f"STORE_NAME.pkl", "wb") as f:
#   pickle.dump(vs, f)
#



from langchain_openai import OpenAIEmbeddings  # Import from langchain_openai instead

import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import pickle
import os
import openai
from langchain.chat_models import ChatOpenAI
import os
os.environ["OPENAI_API_KEY"] = "sk-proj-tjLjSN4Wboyi5uo0ib6BT3BlbkFJD8Gwkjt3oDOXNlXsJoFz"

# Set up OpenAI API key
openai.api_key = "sk-proj-tjLjSN4Wboyi5uo0ib6BT3BlbkFJD8Gwkjt3oDOXNlXsJoFz"

load_dotenv()

st.header('ChatPDF v0.1')
st.sidebar.header(":blue[Welcome to ChatPDF!]")
pdf = st.file_uploader('Upload a PDF file with text in English. PDFs that only contain images will not be recognized.', type=['pdf'])
query = st.text_input('Ask a question about the PDF you entered!', max_chars=300)

if pdf is not None:
    try:
        pdf_doc = PdfReader(pdf)
        txt = ""
        for page in pdf_doc.pages:
            txt += page.extract_text()

        text_split = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # number of characters per chunk
            chunk_overlap=200,  # used to keep the context of a chunk intact with previous and next chunks
            length_function=len
        )
        chunks = text_split.split_text(text=txt)
        embeddings = OpenAIEmbeddings()
        vectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        docs = vectorStore.similarity_search(query=query)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        st.write(response)

        # Saving vector store
        store_name = "vector_store"
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(vectorStore, f)

    except Exception as e:
        st.error(str(e))