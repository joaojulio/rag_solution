import gradio as gr
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from src.langchain_integration.retrieval_qa_chain import (
    build_retrieval_qa_chain, format_query, execute_query_with_cache, clarify_question)

ROLE = "Especialista em Darwin e evolução"
GOAL = "Explicar o conceito de seleção natural conforme descrito no livro"
BACKSTORY = "Você estudou detalhadamente 'On the Origin of Species' e compreende profundamente as teorias de Darwin."

qa_chain = build_retrieval_qa_chain(rerank_k=5, n=10)

def answer_query(user_question: str) -> str:
    """
    Função que recebe a pergunta do usuário, formata o input e retorna a resposta gerada pela chain.
    """
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    clarifying = clarify_question(user_question, openai_api_key)
    if clarifying:
        return (f"Sua pergunta parece ambígua: {clarifying}\n"
                "Por favor, reformule ou adicione mais detalhes à sua pergunta.")
    else:
        combined_query = format_query(ROLE, GOAL, BACKSTORY, user_question)
        result = execute_query_with_cache(qa_chain, combined_query)
        return result.get("answer", "Nenhuma resposta gerada.")

# Cria a interface Gradio
iface = gr.Interface(
    fn=answer_query,
    inputs=gr.components.Textbox(label="Pergunta"),
    outputs=gr.components.Textbox(label="Resposta"),
    title="Advanced RAG Chatbot",
    description="Digite sua pergunta sobre 'On the Origin of Species' de Charles Darwin.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
