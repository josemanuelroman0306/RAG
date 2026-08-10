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


TOP_K = 5

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------
# CONEXIÓN A OPENSEARCH
# ----------------------------------------

client = OpenSearch(
    hosts=[{"host": OS_HOST, "port": OS_PORT}],
    http_auth=(OS_USER, OS_PASS),
    use_ssl=True,
    verify_certs=False
)


# ----------------------------------------
# MODELO DE EMBEDDINGS
# ----------------------------------------

print(f"Usando dispositivo: {DEVICE}")

embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)


def embed_query(text):
    text = "query: " + text
    vec = embedder.encode(text, normalize_embeddings=True)
    return vec.tolist()


# ----------------------------------------
# RECUPERACIÓN VECTORIAL
# ----------------------------------------

def retrieve_chunks(query):

    query_vector = embed_query(query)

    body = {
        "size": TOP_K,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": TOP_K
                }
            }
        }
    }

    resp = client.search(index=INDEX_NAME, body=body)

    sources = []

    for hit in resp["hits"]["hits"]:

        src = hit["_source"]

        sources.append({
            "archivo": src.get("archivo"),
            "chunk_id": src.get("chunk_id")
        })

    return sources


# ----------------------------------------
# EVALUACIÓN
# ----------------------------------------

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"\nPreguntas en dataset: {len(dataset)}")

hit_list = []
precision_list = []
recall_list = []

print("\n=== Evaluando RAG Embeddings ===\n")

for sample in tqdm(dataset):

    query = sample["query"]
    relevant_chunks = sample["relevant_chunks"]
    archivo = sample["archivo"]

    sources = retrieve_chunks(query)

    retrieved_ids = [
        s["chunk_id"] for s in sources
        if s["archivo"] == archivo
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