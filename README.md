# Agente Supermercado

Agente de IA que responde consultas en lenguaje natural sobre un catálogo de productos de supermercado, usando búsqueda semántica (RAG) sobre un archivo CSV.

## 🔗 Demo pública

**URL:** http://144.22.44.19:7860

La app está protegida con autenticación básica.

- **Usuario:** `agosli`
- **Contraseña:** `contrasena_agente`

### Evidencia del deploy funcionando

![Agente funcionando en OCI](./Captura%20de%20pantalla%202026-07-18%20192756.png)

## Arquitectura

El agente sigue un patrón **RAG (Retrieval-Augmented Generation)**:

1. **Carga e indexado**: el catálogo de productos (CSV) se carga con pandas. Se genera un texto de búsqueda combinando Descripción, Marca, Categoría y Subcategoría de cada producto.
2. **Embeddings**: cada fila del catálogo se convierte en un vector numérico usando el modelo `embed-multilingual-v3.0` de Cohere.
3. **Índice vectorial**: los embeddings se almacenan en un índice **FAISS** (búsqueda por similitud, `IndexFlatL2`) para recuperación rápida.
4. **Consulta del usuario**: la pregunta se embeddea de la misma forma, y FAISS devuelve los `top_k` productos más similares semánticamente.
5. **Generación de respuesta**: los productos recuperados se pasan como contexto a un LLM (**Cohere Command R**, vía LangChain), que genera la respuesta en español, con memoria de la conversación.
6. **Interfaz**: **Gradio** expone un chat web simple, con logging de cada interacción (pregunta, productos recuperados, respuesta, tiempo).

```
Usuario → Gradio → Embeddings (Cohere) → Búsqueda FAISS → Contexto + Pregunta → LLM (Cohere) → Respuesta
```

Los embeddings y el índice FAISS se regeneran automáticamente solo si el CSV es más nuevo que los archivos guardados (`embeddings.npy`, `index.faiss`), evitando recalcular en cada arranque si no hubo cambios en los datos.

## Tecnologías

- **Python 3.11**
- **Cohere** (LLM `command-r-08-2024` + Embeddings `embed-multilingual-v3.0`)
- **FAISS** (búsqueda vectorial)
- **LangChain / langchain-cohere** (orquestación del LLM)
- **Gradio** (interfaz de chat web)
- **Pandas / NumPy** (procesamiento de datos)
- **Docker / Podman** (contenedorización)
- **Oracle Cloud Infrastructure (OCI)** (deploy — Compute Instance + Container)

## Despliegue (OCI)

El proyecto está desplegado en una **Compute Instance** de OCI (Oracle Linux 9), corriendo en un contenedor Podman (compatible con Docker CLI).

**Infraestructura:**
- VCN con subnet pública
- Compute Instance (VM.Standard.E5.Flex)
- IP pública reservada (fija): `144.22.44.19`
- Puerto 7860 habilitado en la Security List (OCI) y en el firewall interno (`firewalld`)
- Autenticación básica a nivel de aplicación (Gradio `auth`)

## Configuración local

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/agosli-oss/Agente_supermercado.git
   cd Agente_supermercado
   ```
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Copiar `.env.example` a `.env` y completar las variables (ver abajo)
4. Correr la app:
   ```bash
   python agente.py
   ```
5. Abrir `http://localhost:7860` en el navegador

## Despliegue con Docker/Podman

```bash
docker build -t agente-supermercado .
docker run -d --name agente --restart unless-stopped -p 7860:7860 --env-file .env agente-supermercado
```

## Variables de entorno

| Variable | Descripción |
|---|---|
| `COHERE_API_KEY` | API key de Cohere |
| `CSV_PATH` | Ruta al archivo CSV de productos (default: `archivo.csv`) |
| `AUTH_USER` | Usuario para el acceso a la app |
| `AUTH_PASS` | Contraseña para el acceso a la app |

> ⚠️ El archivo `archivo.csv` no está incluido en el repositorio (excluido por `.gitignore`, ya que contiene datos del catálogo). Debe agregarse manualmente en el mismo directorio que `agente.py`.

## Ejemplos de preguntas y respuestas

**Pregunta:** ¿Qué precio tiene el arroz integral de 1kg de la marca Costeño?
**Respuesta:** El arroz integral de 1kg de la marca Costeño tiene un precio de venta unitario de 8,9 unidades monetarias.

**Pregunta:** ¿Qué stock hay de Pasta Espagueti 500g La Moderna?
**Respuesta:** El stock actual de Pasta Espagueti 500g de la marca La Moderna es de 150 unidades.

**Pregunta:** ¿Cuál es el SKU de Café Tostado y Molido 500g Juan Valdez?
**Respuesta:** El SKU del Café Tostado y Molido 500g de la marca Juan Valdez es MER-031.

## Estructura del repositorio

```
├── agente.py          # Lógica del agente (embeddings, FAISS, LangChain, Gradio)
├── Dockerfile          # Imagen para despliegue en contenedor
├── requirements.txt    # Dependencias de Python
├── .env.example        # Plantilla de variables de entorno
├── .gitignore
└── README.md
```
