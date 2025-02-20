from langchain.prompts import PromptTemplate

def get_custom_prompt_template():
    """
    Retorna um PromptTemplate customizado que utiliza os elementos:
    role, goal, backstory, user_question na variável input e context.
    """
    template = """
Você é um assistente especialista em "On the Origin of Species" de Darwin.

{input}

Contexto:
{context}

Forneça uma resposta detalhada e coerente com base no contexto acima.
"""
    return PromptTemplate(
        input_variables=["input", "context"],
        template=template
    )