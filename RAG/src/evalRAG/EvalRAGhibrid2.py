import os
import json
import numpy as np
from tqdm import tqdm
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
import torch
import warnings
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

# ----------------------------------------
# CONFIGURACIÓN
# ----------------------------------------

OS_HOST = "localhost"
OS_PORT = 9200
OS_USER = "admin"
OS_PASS = "Tu Contraseña"

INDEX_NAME = "iniciativas_chunks2_emb"

DATASET_FILE = "PreguntasEvalRAG.json"


TOP_K = 3
RRF_K = 60

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------
# CONEXIÓN OPENSEARCH
# ----------------------------------------

client = OpenSearch(
    hosts=[{"host": OS_HOST, "port": OS_PORT}],
    http_auth=(OS_USER, OS_PASS),
    use_ssl=True,
    verify_certs=False
)


# ----------------------------------------
# MODELO EMBEDDINGS
# ----------------------------------------

print(f"Usando dispositivo: {DEVICE}")

embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

def embed_query(text):
    text = "query: " + text
    vec = embedder.encode(text, normalize_embeddings=True)
    return vec.tolist()


# ----------------------------------------
# BM25
# ----------------------------------------

def retrieve_bm25(query, size=20):

    body = {
        "size": size,
        "query": {
            "match": {
                "texto": {
                    "query": query
                }
            }
        }
    }

    resp = client.search(index=INDEX_NAME, body=body)

    return resp["hits"]["hits"]


# ----------------------------------------
# kNN
# ----------------------------------------

def retrieve_knn(query, size=20):

    query_vector = embed_query(query)

    body = {
        "size": size,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": size
                }
            }
        }
    }

    resp = client.search(index=INDEX_NAME, body=body)

    return resp["hits"]["hits"]


# ----------------------------------------
# RRF FUSION
# ----------------------------------------

def reciprocal_rank_fusion(bm25_hits, knn_hits):

    scores = {}
    doc_info = {}

    for hit in bm25_hits + knn_hits:

        doc_id = hit["_id"]
        src = hit["_source"]

        doc_info[doc_id] = {
            "archivo": src.get("archivo"),
            "chunk_id": src.get("chunk_id")
        }

    for rank, hit in enumerate(bm25_hits):

        doc_id = hit["_id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (RRF_K + rank + 1)

    for rank, hit in enumerate(knn_hits):

        doc_id = hit["_id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (RRF_K + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []

    for doc_id, score in ranked[:TOP_K]:

        info = doc_info[doc_id]

        results.append({
            "archivo": info["archivo"],
            "chunk_id": info["chunk_id"]
        })

    return results


# ----------------------------------------
# CARGAR DATASET
# ----------------------------------------

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"\nPreguntas en dataset: {len(dataset)}")


# ----------------------------------------
# EVALUACIÓN
# ----------------------------------------

hit_list = []
precision_list = []
recall_list = []

print("\n=== Evaluando RAG Hybrid RRF ===\n")

for sample in tqdm(dataset):

    query = sample["query"]
    relevant_chunks = sample["relevant_chunks"]
    archivo = sample["archivo"]

    bm25_hits = retrieve_bm25(query)
    knn_hits = retrieve_knn(query)

    retrieved = reciprocal_rank_fusion(bm25_hits, knn_hits)

    retrieved_ids = [
        r["chunk_id"] for r in retrieved
        if r["archivo"] == archivo
    ]

    relevant_set = set(relevant_chunks)
    retrieved_set = set(retrieved_ids)

    hit = int(len(relevant_set.intersection(retrieved_set)) > 0)

    precision = len(relevant_set.intersection(retrieved_set)) / TOP_K

    recall = len(relevant_set.intersection(retrieved_set)) / len(relevant_set)

    hit_list.append(hit)
    precision_list.append(precision)
    recall_list.append(recall)


# ----------------------------------------
# RESULTADOS
# ----------------------------------------

hit_rate = np.mean(hit_list)
precision_k = np.mean(precision_list)
recall_k = np.mean(recall_list)

print("\n===== RESULTADOS =====\n")

print(f"HitRate@{TOP_K}: {hit_rate:.3f}")
print(f"Precision@{TOP_K}: {precision_k:.3f}")
print(f"Recall@{TOP_K}: {recall_k:.3f}")


# ----------------------------------------
# GUARDAR RESULTADOS
# ----------------------------------------

results = {
    "HitRate@5": float(hit_rate),
    "Precision@5": float(precision_k),
    "Recall@5": float(recall_k)
}

with open("resultados_rag_rrf.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("\nResultados guardados en resultados_rag_rrf.json")