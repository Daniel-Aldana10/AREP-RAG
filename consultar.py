import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

# Cargar variables de entorno
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "mi-base-conocimiento"

# Inicializar componentes
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY
)

model = ChatOpenAI(
    model="gpt-4o-mini",
    request_timeout=30,
    max_retries=2,
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)


@tool
def buscar_contexto(consulta: str) -> str:
    """Busca información relevante en la base de conocimiento sobre microservicios, IA, seguridad, bases de datos, cloud computing, frontend, DevOps, Python, APIs, blockchain y desarrollo móvil."""
    try:
        documentos = vector_store.similarity_search(consulta, k=3)
        contenido = "\n\n".join(
            f"Fuente: {doc.metadata.get('titulo', 'Sin título')}\n"
            f"Categoría: {doc.metadata.get('categoria', 'N/A')}\n"
            f"Autor: {doc.metadata.get('autor', 'N/A')}\n"
            f"Contenido: {doc.page_content[:500]}..."  # Limitar longitud para no sobrecargar
            for doc in documentos
        )
        return contenido
    except Exception as e:
        return f"Error al buscar en la base de conocimiento: {str(e)}"


# Crear agente
tools = [buscar_contexto]

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un asistente especializado en tecnología con acceso a una base de conocimiento técnica. "
     "Usa la herramienta de búsqueda para encontrar información relevante sobre: "
     "arquitectura de software, inteligencia artificial, seguridad informática, bases de datos, "
     "cloud computing, desarrollo frontend, DevOps, Python para data science, diseño de APIs, "
     "blockchain y desarrollo móvil. Responde de manera clara y concisa en español."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def hacer_pregunta(pregunta: str):
    """Realiza una pregunta al agente RAG"""
    print(f"\n Pregunta: {pregunta}\n")
    print(" Respuesta:\n")

    try:
        respuesta = agent_executor.invoke({"input": pregunta})
        print(respuesta["output"])
    except Exception as e:
        print(f"Error al procesar la pregunta: {str(e)}")


if __name__ == "__main__":
    print("=" * 60)
    print(" Sistema RAG con Pinecone y ChatGPT")
    print(" Especializado en documentación técnica")
    print("=" * 60)
    print("\nTemas disponibles: microservicios, IA, seguridad, bases de datos,")
    print("cloud computing, frontend, DevOps, Python, APIs, blockchain, móvil")

    # Modo interactivo
    while True:
        pregunta = input("\n Tu pregunta (o 'salir' para terminar): ")

        if pregunta.lower() in ['salir', 'exit', 'quit']:
            print("\n ¡Hasta luego!")
            break

        if pregunta.strip():
            hacer_pregunta(pregunta)