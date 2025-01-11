from langchain_community.document_loaders.directory import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_community.document_loaders import DataFrameLoader
import pandas as pd
import logging
from typing import Generic, Iterator, Sequence, TypeVar
from langchain.schema import Document
from langchain_core.stores import BaseStore
from sqlalchemy.orm import sessionmaker, scoped_session
import torch
from sqlalchemy import Column, String, create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from pydantic import BaseModel, Field
from typing import Optional
import os
from dotenv import load_dotenv
from operator import le
import streamlit as st
import torch
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import logging
from typing import Generic, Iterator, Sequence, TypeVar
from langchain.schema import Document
from langchain_core.stores import BaseStore
from sqlalchemy.orm import sessionmaker, scoped_session
from langchain.retrievers import ParentDocumentRetriever
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from semantic_router import SemanticRouter, Route
from semantic_router.sample import chatbotSample, chitchatSample
from langgraph.graph import START, StateGraph
from pydantic import BaseModel, Field
from typing import Literal
from grader import GradeDocuments
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
from sqlalchemy import Column, String, create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
from sqlalchemy import create_engine
import io

load_dotenv()

device = "cuda" if torch.cuda.is_available() else  "cpu"
Base = declarative_base()

class DocumentModel(BaseModel):
    key: Optional[str] = Field(None)
    page_content: Optional[str] = Field(None)
    metadata: dict = Field(default_factory=dict)

class SQLDocument(Base):
    __tablename__ = "docstore"
    key = Column(String, primary_key=True)
    value = Column(JSONB)

    def __repr__(self):
        return f"<SQLDocument(key='{self.key}', value='{self.value}')>"

logger = logging.getLogger(__name__)

D = TypeVar("D", bound=Document)

class PostgresStore(BaseStore[str, DocumentModel], Generic[D]):
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)
        Base.metadata.create_all(self.engine)
        self.Session = scoped_session(sessionmaker(bind=self.engine))

    def serialize_document(self, doc: Document) -> dict:
        return {"page_content": doc.page_content, "metadata": doc.metadata}

    def deserialize_document(self, value: dict) -> Document:
        return Document(page_content=value.get("page_content", ""), metadata=value.get("metadata", {}))


    def mget(self, keys: Sequence[str]) -> list[Document]:
        with self.Session() as session:
            try:
                sql_documents = session.query(SQLDocument).filter(SQLDocument.key.in_(keys)).all()
                return [self.deserialize_document(sql_doc.value) for sql_doc in sql_documents]
            except Exception as e:
                logger.error(f"Error in mget: {e}")
                session.rollback()
                return [] 
    def mset(self, key_value_pairs: Sequence[tuple[str, Document]]) -> None:
        with self.Session() as session:
            try:
                serialized_docs = []
                for key, document in key_value_pairs:
                    serialized_doc = self.serialize_document(document)
                    serialized_docs.append((key, serialized_doc))

                documents_to_update = [SQLDocument(key=key, value=value) for key, value in serialized_docs]
                session.bulk_save_objects(documents_to_update, update_changed_only=True)
                session.commit()
            except Exception as e:
                logger.error(f"Error in mset: {e}")
                session.rollback()


    def mdelete(self, keys: Sequence[str]) -> None:
        with self.Session() as session:
            try:
                session.query(SQLDocument).filter(SQLDocument.key.in_(keys)).delete(synchronize_session=False)
                session.commit()
            except Exception as e:
                logger.error(f"Error in mdelete: {e}")
                session.rollback() 
    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        with self.Session() as session:
            try:
                query = session.query(SQLDocument.key)
                if prefix:
                    query = query.filter(SQLDocument.key.like(f"{prefix}%"))
                for key in query:
                    yield key[0]
            except Exception as e:
                logger.error(f"Error in yield_keys: {e}")
                session.rollback()

embedding = HuggingFaceEmbeddings(
    model_name='hiieu/halong_embedding',
    model_kwargs={'device': device},
    encode_kwargs= {'normalize_embeddings': False}
)

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=8096*2)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=512)

QDRANT_API = os.getenv("QDRANT_API_KEY")
QDRANT_URL = "https://21a75178-7457-4e63-974b-666f8174af84.us-west-2-0.aws.cloud.qdrant.io:6333"
QDRANT_COLLECTION = 'corpus_tvpl'

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API
)

DB_URL = "localhost:5432"

# Thông tin PostgreSQL
DB_NAME = 'parent'
DB_USER = 'postgres'
DB_PASSWORD = os.getenv("DB_PASSWORD")

URL_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_URL}/{DB_NAME}"
document_store = PostgresStore(URL_STRING)

vector_store = QdrantVectorStore(
    client=client, 
    collection_name='corpus_tvpl',
    embedding=embedding
)

parent_document_retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=document_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_Key")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")    
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")