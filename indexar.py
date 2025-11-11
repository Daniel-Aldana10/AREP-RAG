import json
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

# Configuración
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "mi-base-conocimiento"


def cargar_documentos_json(ruta_json):
    """Carga documentos desde un archivo JSON"""
    with open(ruta_json, 'r', encoding='utf-8') as f:
        datos = json.load(f)

    documentos = []
    for item in datos:
        # Construir contenido completo del documento
        contenido_completo = f"Título: {item['titulo']}\n\n{item['contenido']}"

        doc = Document(
            page_content=contenido_completo,
            metadata={
                "id": item["id"],
                "titulo": item["titulo"],
                "categoria": item["metadata"]["categoria"],
                "subcategoria": item["metadata"]["subcategoria"],
                "fecha": item["metadata"]["fecha"],
                "autor": item["metadata"]["autor"],
                "nivel": item["metadata"]["nivel"],
                "tags": ", ".join(item["metadata"]["tags"]),
                "idioma": item["metadata"]["idioma"],
                "tiempo_lectura": item["metadata"]["tiempo_lectura"]
            }
        )
        documentos.append(doc)

    return documentos


def crear_indice_pinecone():
    """Crea o verifica el índice en Pinecone"""
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Verificar si el índice existe
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creando índice '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,  # text-embedding-3-small usa 1536 dimensiones
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Índice '{INDEX_NAME}' creado exitosamente")
    else:
        print(f"El índice '{INDEX_NAME}' ya existe")


def indexar_documentos(ruta_json):
    """Indexa documentos en Pinecone"""
    print("\nCargando documentos desde JSON...")
    documentos = cargar_documentos_json(ruta_json)
    print(f" {len(documentos)} documentos cargados")

    print("\nDividiendo documentos en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"{len(chunks)} chunks creados")

    print("\n🌲 Configurando Pinecone...")
    crear_indice_pinecone()

    print("\nInicializando modelo de embeddings...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )

    print("\nIndexando documentos en Pinecone...")
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    print("\n¡Indexación completada exitosamente!")
    return vector_store


if __name__ == "__main__":
    ruta_json = "data/documentos.json"

    # Verificar que el archivo existe
    if not os.path.exists(ruta_json):
        print(f"Error: No se encuentra el archivo {ruta_json}")
        print("Asegúrate de que el archivo JSON existe en la ruta especificada")
        exit(1)

    vector_store = indexar_documentos(ruta_json)
    print(f"\nTu base de conocimiento está lista en Pinecone (índice: {INDEX_NAME})")