# Document Q&A API with Vector Search


## Overview
Your task is to build a simple Retrieval-Augmented Generation (RAG) service in Python. This service will answer questions based on a small, provided corpus of public domain text documents. The goal is to demonstrate your ability to design, build, and test a system that integrates data processing, storage, and a machine learning component within a service-oriented architecture.

The core of the system involves two main functions: data ingestion and querying. The ingestion process will read text documents, split them into manageable chunks, generate vector embeddings for each chunk using a pre-trained model, and store these embeddings in a suitable vector store. The querying function will be exposed via a simple HTTP API. This API will take a user's question, embed it, perform a similarity search against the vector store to find the most relevant document chunks, and then synthesize an answer.

We are most interested in your design choices, the clarity and quality of your code, and your approach to testing. The specific technologies you choose for the vector store or API framework are less important than your ability to justify your decisions in the README. A good solution will be well-structured, easy to run, and include thoughtful trade-off discussions. The final answer generation can be a simple concatenation of the retrieved chunks or a mocked response; a full LLM integration is not required.

## Deliverables
- A `README.md` file explaining your design choices, trade-offs, and clear instructions on how to set up dependencies, run the ingestion process, and start the API server.
- An ingestion mechanism (e.g., a command-line script or API) that processes source text files and populates your chosen vector store.
- A web API (e.g., using FastAPI) with at least one endpoint that accepts a question and returns a relevant answer based on the ingested documents.
- Simple UI is optional but not required; the focus is on the backend service.
- Automated tests for the core logic of your application.

## Suggested tools / libraries
- API Framework: FastAPI
- Embedding Model: A model from the `sentence-transformers` library
- Vector Storage: ElasticSearch.
- Containerization: Using Docker to package the application.

## On AI assistants & follow-up
- Make sure to capture logs or audit on your working process and share with us how you used AI assistants, if at all. We are interested in your thought process and how you approached the problem, including any tools you used to assist you.
- Be prepared to walk us through any part of your submitted code and justify your design decisions during the follow-up interview.
- The goal is to evaluate your engineering judgment. We value a simple, well-executed solution over a complex one with many features.
- Your submission must be your own work, although you may use open-source libraries and consult public documentation.

