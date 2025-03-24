import streamlit as st
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain.retrievers import ParentDocumentRetriever
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from semantic_router import SemanticRouter, Route
from semantic_router.sample import chatbotSample, chitchatSample
from grader import GradeDocuments
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from parent_document_retriever import PostgresStore


load_dotenv()
store = LocalFileStore("./cache/")
# Define environment variables
NGROK_URL = "0.tcp.ap.ngrok.io:17032"
PG_URL = "localhost:5432"
DB_NAME = 'parent'
DB_USER = 'postgres'
DB_PASSWORD = os.getenv("DB_PASSWORD")
# URL_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{NGROK_URL}/{DB_NAME}"
URL_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{PG_URL}/{DB_NAME}"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_Key")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")    
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GPT_API_KEY = os.getenv("GPT_API_KEY")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Initialize embedding model --- #
@st.cache_resource
def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name='hiieu/halong_embedding',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        store,
        namespace="hiieu/halong_embedding"
    )
    return embedder

embedding = get_embeddings()

# --- Initialize Qdrant client and vector store --- #
@st.cache_resource
def get_vectorstore():
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    
    return QdrantVectorStore(
        client=client, 
        collection_name="corpus_tvpl",
        embedding=embedding
    )

@st.cache_resource
def get_parent_doc_retrieve(_vector_store, _document_store, _child_splitter, _parent_splitter):
    search_kwargs = {"k": 5}
    parent_document_retriever = ParentDocumentRetriever(
        vectorstore=_vector_store,
        docstore=_document_store,
        child_splitter=_child_splitter,
        parent_splitter=_parent_splitter,
        search_kwargs=search_kwargs
    )
    return parent_document_retriever

# --- Initialize LLM --- #
@st.cache_resource
def get_llm():
    return ChatOpenAI(
        api_key = GPT_API_KEY,
        temperature=0,
        model="gpt-4o-mini"
    )
    # return ChatGroq(
    #     temperature=0,
    #     groq_api_key=GROQ_API_KEY,
    #     model_name="llama-3.1-8b-instant"
    # )

# --- Create prompt template --- #
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Bạn là một trợ lý ảo nhiệt tình và thành thật tư vấn về luật pháp.
            Bạn sẽ dựa trên các thông tin được cung cấp để trả lời các câu hỏi của người dùng.
            #Nếu không biết câu trả lời, bạn phải trả lời là tôi không biết.
            Dựa vào thông tin sau trả lời câu hỏi:
            {context}
            """
            ,
        ),
        ("human", "{input}"),
    ]
)

# Initialize components
vector_store = get_vectorstore()
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=8096*2)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=512)
document_store = PostgresStore(URL_STRING)
parent_retriever = get_parent_doc_retrieve(vector_store, document_store, child_splitter, parent_splitter)

llm = get_llm()

LEGAL_ROUTE_NAME = 'legal'
CHITCHAT_ROUTE_NAME = 'normal'

legalRoute = Route(name=LEGAL_ROUTE_NAME, samples=chatbotSample)
chitchatRoute = Route(name=CHITCHAT_ROUTE_NAME, samples=chitchatSample)
semanticRouter = SemanticRouter(embedding, routes=[legalRoute, chitchatRoute])

# Streamlit UI
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 25% !important; 
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 Legal Assistant")
st.caption("Ask me anything about Vietnamese law!")  

with st.sidebar:
    st.write("## Instructions")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if user_question := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)
    result_response = ""        
    with st.chat_message("assistant"):
        if user_question:
            guidedRoute = semanticRouter.guide(user_question)[1]
            st.caption(F"Normal Query: {guidedRoute == CHITCHAT_ROUTE_NAME}")

            if guidedRoute == LEGAL_ROUTE_NAME:        
                with st.spinner("Searching for relevant information..."):
                    docs = parent_retriever.invoke(user_question)
                    
                    # Display sources
                    with st.sidebar:
                        st.write(f"#### Num of original documents: {len(docs)}" )
                        for i, doc in enumerate(docs, 1):
                            with st.expander(f"Sources {i}: {doc.metadata['title'][:40]}..."):
                                st.write(doc.page_content)
                                
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    messages = prompt.invoke({"input": user_question, "context": context})
                    with st.spinner("Generating answer..."):
                        response = llm.invoke(messages)
                    
            elif guidedRoute == CHITCHAT_ROUTE_NAME:
                    messages = prompt.invoke({"input": user_question, "context": ""})
                    with st.spinner("Generating answer..."):
                        response = llm.invoke(messages)

            result_response = response.content
            st.write(result_response)
    st.session_state.messages.append({"role": "assistant", "content": result_response})