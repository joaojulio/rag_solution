import os
from pinecone import Pinecone
from langchain_pinecone import Pinecone as LC_Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.rag_pipeline.prompt_engineering.prompt_templates import get_custom_prompt_template
from src.rag_pipeline.re_ranker import ReRankRetriever
import diskcache as dc
import hashlib
import json
from dotenv import load_dotenv


cache = dc.Cache("cache_respostas")

def load_keys():
    load_dotenv()
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("A variável de ambiente OPENAI_API_KEY não está definida.")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("A variável de ambiente PINECONE_API_KEY não está definida.")
    
    return openai_api_key, pinecone_api_key

def build_retrieval_qa_chain(index_name="origin-of-species-index", rerank_k=5, n=10):
    """
    Constrói e retorna uma chain RetrievalQA utilizando LangChain com re-ranking.
    """
    openai_api_key, pinecone_api_key = load_keys()
    
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    vectorstore = LC_Pinecone.from_existing_index(index_name, embeddings, text_key="text")
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": n})

    rerank_retriever = ReRankRetriever(base_retriever=base_retriever, rerank_k=rerank_k, n=n)

    llm = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", temperature=0)
    custom_prompt = get_custom_prompt_template()

    combine_docs_chain = create_stuff_documents_chain(llm, custom_prompt)
    qa_chain = create_retrieval_chain(rerank_retriever, combine_docs_chain)

    return qa_chain

def format_query(role, goal, backstory, user_question):
    """
    Combina as informações de role, goal, backstory e a pergunta do usuário em uma única string.
    """
    return f"Role: {role}\nGoal: {goal}\nBackstory: {backstory}\nQuestion: {user_question}"

def get_cache_key(params: dict) -> str:
    """
    Gera uma chave de cache única a partir dos parâmetros da consulta.
    """
    key_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()

def execute_query_with_cache(qa_chain, combined_query):
    """
    Verifica se a resposta para os parâmetros fornecidos já está no cache.
    Se sim, retorna a resposta cacheada. Se não, executa a chain e armazena o resultado no cache.
    """
    query_params = {"input": combined_query}
    cache_key = get_cache_key(query_params)
    
    if cache_key in cache:
        return cache[cache_key]
    
    result = qa_chain.invoke(query_params)
    cache[cache_key] = result
    return result

def clarify_question(user_question, openai_api_key):
    """
    Usa o LLM para determinar se a pergunta é ambígua e, se for, gera uma pergunta de clarificação.
    """
    clarifying_prompt = (
        f"Analise a seguinte pergunta: '{user_question}'. "
        "Se a pergunta for ambígua ou complexa, formule uma pergunta de clarificação para que o usuário possa fornecer mais detalhes. "
        "Caso contrário, responda apenas 'OK'."
    )
    clarifier = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", temperature=0, openai_api_key=openai_api_key)
    response = clarifier.invoke(clarifying_prompt)
    output = response.content.strip()
    if output.upper() != "OK":
        return output
    return None

def main():
    openai_api_key, _ = load_keys()
    qa_chain = build_retrieval_qa_chain(rerank_k=5, n=10)
    
    role = "Especialista em Darwin e evolução"
    goal = "Explicar o conceito de seleção natural conforme descrito no livro"
    backstory = "Você estudou detalhadamente 'On the Origin of Species' e compreende profundamente as teorias de Darwin."
    
    user_question = input("Digite sua pergunta: ")

    clarifying_question = clarify_question(user_question, openai_api_key)
    if clarifying_question:
        print("\nA pergunta parece ambígua. Pergunta de clarificação:")
        print(clarifying_question)
        additional_info = input("Por favor, forneça mais detalhes: ")
        user_question = f"{user_question}\nDetalhes adicionais: {additional_info}"

    combined_query = format_query(role, goal, backstory, user_question)
    
    result = execute_query_with_cache(qa_chain, combined_query)
    
    print("\nResposta:")
    print(result["answer"])
    
if __name__ == "__main__":
    main()
