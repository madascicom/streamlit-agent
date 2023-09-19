import os
import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Workshop AI: Chat cu propriile documente", page_icon="🤖")
st.title("Chat cu propriile documente")
st.markdown("""
Interfață pentru participanții la <a href=https://comunicarestiintifica.ro/workshop-ai-module-avansate/ target=_blank>Workshopul "AI cu propriile documente".</a>

Nu uita: Această aplicație este utilă pentru a afla detalii din pdf-urile tale, nu pentru sumarizare.

Aplicația îți arată și sursele din care a dedus răspunsul, așa că dacă ai dubii, poți verifica adevărul.
""", unsafe_allow_html=True)


@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Surse răspuns")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.container.write(f"**Document {idx} from {source}**")
            self.container.markdown(doc.page_content)


huggingfacehub_api_token = st.sidebar.text_input("token_personal", type="password")
if not huggingfacehub_api_token:
    st.info("Pasul 1: Te rog adaugă token_personal obținut la workshop în bara din stânga (pe laptop). Pe mobil, apasă pe semnul > din stânga sus.")
    st.stop()

uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("Pasul 2: Încarcă unul sau mai multe PDF-uri")
    st.stop()

retriever = configure_retriever(uploaded_files)

# Setup memory for contextual conversation
#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Setup LLM and QA chain
llm = HuggingFaceHub(repo_id="declare-lab/flan-alpaca-large", huggingfacehub_api_token=huggingfacehub_api_token, model_kwargs={"temperature":0, "max_length":512})
qa_chain = RetrievalQA.from_llm(
    llm, retriever=retriever, verbose=True
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Adresează o întrebare!")
real_query = f"{user_query} Give as much context about the answer as possible. If you cannot find the answer in the documents, say that you don't know the answer."

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(real_query, callbacks=[stream_handler, retrieval_handler])
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
