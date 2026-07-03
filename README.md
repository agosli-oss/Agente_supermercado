# Agente Supermercado

Agente de IA que responde consultas en lenguaje natural sobre un catálogo de productos.

## Tecnologías
- Python
- Cohere (LLM + Embeddings)
- FAISS (búsqueda vectorial)
- LangChain
- Gradio

## Configuración
1. Clonar el repositorio
2. Instalar dependencias: `pip install -r requirements.txt`
3. Copiar `.env.example` a `.env` y completar con tu API key de Cohere
4. Correr el notebook

## Variables de entorno
- `COHERE_API_KEY`: API key de Cohere
- `CSV_PATH`: ruta al archivo CSV de productos
