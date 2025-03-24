# legal-chatbot

## Project Overview

This legal chatbot processes and analyzes legal documents using natural language processing and retrieval augmented generation (RAG) techniques. It features hierarchical document handling (parent-child relationships), semantic routing, and document grading capabilities to provide accurate responses to legal queries.

## Technologies

- **Language**: Python 3.11+
- **Database**: PostgreSQL for document storage and retrieval
- **Vector Store**: For semantic document embeddings
- **LLM Integration**: Via LangGraph
- **Document Processing**: Custom parent-child document retrieval system
- **Semantic Routing**: For directing queries to appropriate processing pipelines

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/legal-chatbot.git
   cd legal-chatbot
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env  # If available
   # Edit .env file to configure database connection and API keys
   ```

## Project Structure

- **app.py**: Main application entry point
- **parent_document_retriever/**: Handles hierarchical document relationships
  - `DocumentModel.py`: Document schema and data models
  - `clear_tables.py`: Database maintenance utilities
  - `inspect_db.py`: Database inspection tools
- **grader/**: Document quality assessment system
  - `grade_documents.py`: Evaluates document quality and relevance
- **semantic_router/**: Routes queries to appropriate processing pipelines
- **cache/**: Stores cached results to improve performance
- **ipynb/**: Jupyter notebooks for development and demonstration
  - `create_vectordb.ipynb`: Vector database creation
  - `rag_langgraph.ipynb`: RAG implementation with LangGraph
  - `parent-child-retrieve.ipynb`: Document hierarchy retrieval examples

## Pipeline

The system works through the following pipeline:

1. **Document Ingestion**: Legal documents are processed and stored in PostgreSQL
2. **Vectorization**: Documents are embedded into vector representations
3. **Semantic Indexing**: Documents are indexed for efficient retrieval
4. **Query Processing**: User queries are analyzed using semantic routing
5. **Document Retrieval**: Relevant documents are retrieved using parent-child relationships
6. **Document Grading**: Retrieved documents are graded for relevance
7. **Response Generation**: Final response is generated using RAG techniques

## Running the Application

### Basic Usage

```bash
python app.py
```

### Using Jupyter Notebooks

Navigate to specific notebooks in the `ipynb/` directory:
- `main.ipynb` - General overview and examples
- `rag_langgraph.ipynb` - RAG implementation examples
- `parent-child-retrieve.ipynb` - Document hierarchy retrieval examples
- `create_vectordb.ipynb` - Vector database creation and management

## Database Management

### Inspecting the Database

To view the current state of the database:

```bash
python parent_document_retriever/inspect_db.py
```

### Clearing Database Tables

To reset database tables (use with caution):

```bash
python parent_document_retriever/clear_tables.py
```

You can clear specific tables:

```bash
python parent_document_retriever/clear_tables.py --tables documents,embeddings
```

## License

See the [LICENSE](LICENSE) file for details.