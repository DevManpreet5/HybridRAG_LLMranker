import os
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
from utils.completion import complete  

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=".chroma_store", embedding_function=embedding_model)

class HybridRetriever:
    def __init__(self):
        self.bm25 = None
        self.texts = []

    def index(self, chunks):
        self.texts = chunks
        tokenized = [text.lower().split() for text in chunks]
        self.bm25 = BM25Okapi(tokenized)
        docs = [Document(page_content=chunk, metadata={"id": i}) for i, chunk in enumerate(chunks)]
        vectorstore.add_documents(docs)

    def rerank_with_llm(self, query, candidates):
        scored = []
        for text in candidates:
            prompt = f"""
You're a smart retrieval system. Score the relevance of the following context to the query on a scale of 0 to 10. Only return a number.

Query: {query}
Context: {text}
Relevance Score (0–10):
"""
            try:
                score_str = complete(prompt).strip()
                score = float(score_str.split()[0])
                scored.append((text, score))
            except:
                scored.append((text, 0))  
        reranked = sorted(scored, key=lambda x: x[1], reverse=True)
        return reranked

    def retrieve(self, query, k=5):
        bm25_hits = self.bm25.get_top_n(query.lower().split(), self.texts, n=k)
        docs = vectorstore.similarity_search(query, k=k)
        vector_hits = [doc.page_content for doc in docs]
        candidates = list(set(bm25_hits + vector_hits))
        reranked = self.rerank_with_llm(query, candidates)
        return reranked[:k]
    

retriever = HybridRetriever()
