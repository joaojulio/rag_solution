from langchain.schema import BaseRetriever, Document
from pydantic import BaseModel, Field, Extra
from sentence_transformers import CrossEncoder

class ReRankRetriever(BaseRetriever, BaseModel):
    base_retriever: BaseRetriever
    rerank_k: int = Field(default=5)
    n: int = Field(default=10)
    cross_encoder: CrossEncoder = Field(default_factory=lambda: CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"))

    def get_relevant_documents(self, query: str):
        # Recupera inicialmente n documentos do retriever base
        docs = self.base_retriever.invoke(query)
        if len(docs) > self.n:
            docs = docs[:self.n]
        # Cria pares (query, documento) para o re-ranking
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.cross_encoder.predict(pairs)
        # Ordena os documentos de forma decrescente pelo score
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        ranked_docs = [doc for doc, score in ranked]
        return ranked_docs[:self.rerank_k]

    @property
    def _chain_type(self) -> str:
        return "rerank_retriever"

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True
