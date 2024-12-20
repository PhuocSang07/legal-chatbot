from rag.state import State

class RAG():
    def __init__(self, embedding, url, api_key, collection_name = "qa_tvpl"):
        self.embedding = embedding
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name

        self.vectorstore = get_vectorstore()

    @st.cache_resource
    def get_vectorstore(self):
        client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
        )
        
        return QdrantVectorStore(
            client=client, 
            collection_name=self.collection_name,
            embedding=self.embedding
        )

    def retrieve(self, state: State, k = 5):
        retrieved_docs = self.vectorstore.similarity_search(state["question"], k=k)
        return {"context": retrieved_docs}