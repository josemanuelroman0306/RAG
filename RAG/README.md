# Sistema RAG mediante LLMs

Sistema de **Retrieval-Augmented Generation (RAG)** desarrollado como parte del Trabajo Fin de Grado del Grado en Ingeniería Informática, con el objetivo de estudiar la recuperación de información y la generación de respuestas mediante modelos de lenguaje de gran tamaño.

El sistema permite realizar consultas sobre un corpus de documentos parlamentarios, recuperando previamente los fragmentos más relevantes y utilizándolos como contexto para generar la respuesta mediante un LLM.

## Descripción

El proyecto implementa un pipeline RAG compuesto por dos etapas principales:

1. **Recuperación de información:** búsqueda de los fragmentos más relevantes del corpus documental.
2. **Generación de la respuesta:** utilización de los fragmentos recuperados como contexto para un modelo de lenguaje.

Se han estudiado diferentes estrategias de recuperación:

- BM25
- Recuperación mediante embeddings
- Búsqueda híbrida de OpenSearch
- Reciprocal Rank Fusion (RRF)

Para la generación de respuestas se utiliza **Mistral-7B-Instruct**, cargado mediante cuantización de 4 bits para reducir los requisitos de memoria.

## Arquitectura

El flujo general del sistema es:

```text
Pregunta del usuario
        │
        ▼
Recuperación de información
        │
        ├── BM25
        ├── Embeddings
        ├── Hybrid OpenSearch
        └── RRF
        │
        ▼
Fragmentos relevantes
        │
        ▼
Construcción del contexto
        │
        ▼
Mistral-7B-Instruct
        │
        ▼
Respuesta generada
```


## Licencia

El código desarrollado para este proyecto se distribuye bajo la licencia MIT.

La memoria del Trabajo Fin de Grado, los datasets, documentos,
modelos y demás recursos externos mantienen sus respectivas
licencias y condiciones de uso.