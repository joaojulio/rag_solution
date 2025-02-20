# Advanced RAG Chatbot

Este projeto é uma implementação de um sistema de **IA Generativa com Retrieval-Augmented Generation (RAG)**, voltado para responder perguntas sobre o livro *On the Origin of Species* de Charles Darwin. O sistema combina diversas etapas – desde a ingestão e pré-processamento do texto, geração de embeddings, indexação em um banco vetorial (Pinecone), recuperação e re-ranking dos chunks relevantes, até a composição do prompt e a geração da resposta final por meio de um LLM. Além disso, o sistema conta com um mecanismo de cache e uma etapa de clarificação para lidar com perguntas ambíguas.

<img src="docs/diagrama_arquitetura.png" alt="Diagrama de Arquitetura" width="300"/>

## Requisitos do Case

- **Linguagem:** Python  
- **Framework:** Utilização de modelos generativos integrados via LangChain (incluindo ChatOpenAI para modelos de chat)  
- **Arquitetura:**
  - *Ingestão e pré-processamento de dados*
  - *Geração de embeddings e indexação no Pinecone*
  - *Recuperação com re-ranking*
  - *Composição de prompt* (usando informações de **role**, **goal**, **backstory** e **question**)
  - *Geração de resposta final*
- **Diferenciais:**
  - Implementação de re-ranking para aprimorar a relevância dos documentos recuperados
  - Mecanismo de clarificação para perguntas ambíguas
  - Cache para otimização de consultas repetidas
- **Demonstração:** Vídeo demonstrativo

## Instalação

1. **Clone o repositório:**

    ```bash
    git clone https://github.com/seu-usuario/advanced_rag.git
    cd advanced_rag
2. **Crie e ative um ambiente virtual (usando Anaconda ou virtualenv):**
    ```bash
    conda create --name advanced_rag python=3.9
    conda activate advanced_rag
3. **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    Observação: Certifique-se de que todas as dependências necessárias (como Pinecone, langchain-openai, langchain-community, Gradio, diskcache, etc.) estejam instaladas conforme especificado no requirements.txt.
4. **Configure as variáveis de ambiente:**
Crie um arquivo .env na raiz do projeto com o seguinte conteúdo (substitua as chaves conforme necessário):
    ```bash
    OPENAI_API_KEY=your_openai_api_key
    PINECONE_API_KEY=your_pinecone_api_key
    PINECONE_ENV=your_pinecone_env  # Ex.: "us-east-1"

## Execução

### Front-End com Gradio
Para iniciar a interface do chatbot, execute:
    ```bash
    python src/front_end/app.py
Isso abrirá a interface Gradio em seu navegador, onde você poderá digitar sua pergunta sobre On the Origin of Species e receber uma resposta gerada pelo sistema.

## Outros Módulos
- **Pré-processamento e Indexação:**
 - Execute o script de preparação de dados (prepare_data.py) para gerar os chunks do texto.
 - Execute o script de indexação (pinecone_indexer.py) para gerar os embeddings e realizar o upsert no Pinecone.
- **Testes:**
    Para rodar a suíte de testes, execute:
    ```bash
    pytest tests/

## Funcionalidades Principais
- **Ingestão e Pré-processamento:**
    O texto do livro é carregado, processado e dividido em chunks para facilitar a recuperação.

- **Geração de Embeddings e Indexação:**
    Cada chunk é convertido em um vetor de embedding utilizando o modelo OpenAI e indexado em um banco vetorial (Pinecone).

- **Recuperação com Re-Ranking:**
    Ao receber uma consulta, o sistema recupera os chunks mais relevantes e os reordena utilizando um modelo cross-encoder.

- **Composição do Prompt e Geração de Resposta:**
    Um template customizado (que combina informações de role, goal, backstory e question) é utilizado para compor o prompt final enviado ao LLM para gerar a resposta.

- **Mecanismo de Cache:**
    Respostas são armazenadas em cache para otimizar consultas repetidas.

- **Etapa de Clarificação:**
    Se a pergunta do usuário for ambígua, o sistema utiliza um LLM para gerar uma pergunta de clarificação e solicita mais detalhes antes de prosseguir.

## Considerações e Evolução do Produto
- **Gerenciamento de Contexto e Memória:**
    Atualmente, o sistema não implementa uma memória de conversação multi-turno, mas essa funcionalidade é recomendada para futuras evoluções.

- **Evolução para Agentic RAG:**
    O Advanced RAG estabelece uma base robusta para respostas contextualizadas. Como próxima etapa, o Agentic RAG pode ser implementado para adicionar uma camada de “raciocínio multi-etapas” ou uso de ferramentas externas. Por exemplo:

    - Recuperação Multi-Hop: O agente poderia recuperar trechos do livro de Darwin e, em seguida, consultar outra base de conhecimento para comparar com outro autor.
    - Clarificação Dinâmica: Poderia fazer perguntas adicionais para definir melhor o contexto (ex.: "Você se refere à edição de 1859 ou de 1872?").
    - Orquestração de Tarefas: O Advanced RAG serve como base, enquanto o Agentic RAG introduzirá um nível extra de interação e raciocínio para tarefas mais complexas.

- **Interface e Usabilidade:**
    O front-end com Gradio demonstra as funcionalidades do sistema. Futuramente, a interface pode ser aprimorada para incluir histórico de conversa, suporte a multi-turnos, entre outros recursos.

- **Recomendações Técnicas:**
 - Migrar para a nova interface de ChatOpenAI conforme as recomendações do LangChain.
 - Considerar a implementação de metadados para referência de fontes (capítulos, seções, etc.) para maior transparência nas respostas.

 ## Demonstração
Um vídeo curto que demonstra o funcionamento do sistema, destacando:

- Interação do Usuário:
    Como o usuário insere sua pergunta na interface Gradio.

- Etapa de Clarificação:
    Se a pergunta for ambígua, o sistema solicita que o usuário forneça mais detalhes.

- Fluxo Completo:
    Desde a ingestão dos dados, recuperação com re-ranking, composição do prompt e geração da resposta final.

- Uso do Cache:
    Demonstração de como consultas repetidas são otimizadas com o mecanismo de cache.

Assista ao vídeo de demonstração no YouTube: (link)