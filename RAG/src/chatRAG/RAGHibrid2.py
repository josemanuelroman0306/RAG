import os
import warnings
import torch
from opensearchpy import OpenSearch  # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer  # type: ignore
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ----------------------------------------
# CONFIGURACIÓN
# ----------------------------------------
OS_HOST = "localhost"
OS_PORT = 9200
OS_USER = "admin"
OS_PASS = "Tu Contraseña"
INDEX_NAME = "iniciativas_chunks2_emb"

TOP_K = 10
RRF_K = 60

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore", category=UserWarning)

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
# MODELO GENERATIVO
# ----------------------------------------
print(f"Usando dispositivo: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    dtype=torch.float16,
    low_cpu_mem_usage=True
)

# ----------------------------------------
# MODELO DE EMBEDDINGS
# ----------------------------------------
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
# FUSIÓN RRF + FUENTES
# ----------------------------------------
def reciprocal_rank_fusion(bm25_hits, knn_hits, top_k=TOP_K):
    scores = {}
    hit_map = {}

    # Guardamos todos los hits
    for hit in bm25_hits + knn_hits:
        hit_map[hit["_id"]] = hit

    # Ranking BM25
    for rank, hit in enumerate(bm25_hits):
        doc_id = hit["_id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (RRF_K + rank + 1)

    # Ranking kNN
    for rank, hit in enumerate(knn_hits):
        doc_id = hit["_id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (RRF_K + rank + 1)

    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    final_chunks = []
    sources = []

    for doc_id, fused_score in ranked_docs[:top_k]:
        doc = client.get(index=INDEX_NAME, id=doc_id)
        src = doc["_source"]

        bloque = f"""
Legislatura: {src.get('legislatura')}
Órgano: {src.get('organo')}
Presidente: {src.get('presidente')}
Fecha: {src.get('dia')} {src.get('mes')} {src.get('anio')}
Tipo sesión: {src.get('tipo_sesion')}
Tipo iniciativa: {src.get('tipo_iniciativa')}
Interviniente: {src.get('interviniente')}

Texto:
{src.get('texto')}
""".strip()

        final_chunks.append(bloque)

        sources.append({
            "archivo": src.get("archivo"),
            "chunk_id": src.get("chunk_id"),
            "legislatura": src.get("legislatura"),
            "fecha": f"{src.get('dia')} {src.get('mes')} {src.get('anio')}",
            "interviniente": src.get("interviniente"),
            "tipo_iniciativa": src.get("tipo_iniciativa"),
            "fused_score": fused_score
        })

    return final_chunks, sources

# ----------------------------------------
# GENERACIÓN
# ----------------------------------------
def generate_answer(query, chunks):
    context = "\n\n".join(chunks)

    prompt = f"""Usando el siguiente contexto de documentos parlamentarios, responde la pregunta de forma clara, precisa y basada únicamente en el contexto proporcionado.

Contexto:
{context}

Pregunta: {query}

Respuesta:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=10048).to(DEVICE)

    output = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = answer.replace(prompt, "").strip()
    return answer

# ----------------------------------------
# BUCLE PRINCIPAL
# ----------------------------------------
print("=== RAG HÍBRIDO MANUAL (BM25 + kNN + RRF) CON FUENTES ===")

while True:
    query = input("\nTú: ")
    if query.lower() in ["exit", "salir", "quit"]:
        break

    print("\nEjecutando BM25...")
    bm25_hits = retrieve_bm25(query)

    print("Ejecutando kNN...")
    knn_hits = retrieve_knn(query)

    print("Fusionando rankings (RRF)...")
    chunks, sources = reciprocal_rank_fusion(bm25_hits, knn_hits)

    print("Generando respuesta...")
    answer = generate_answer(query, chunks)

    print(f"\nBot:\n{answer}")

    print("\nFuentes utilizadas:\n")

    for i, src in enumerate(sources, 1):
        print(f"Fuente {i}:")
        print(f"  Archivo: {src['archivo']}")
        print(f"  Chunk ID: {src['chunk_id']}")
        print(f"  Legislatura: {src['legislatura']}")
        print(f"  Fecha: {src['fecha']}")
        print(f"  Interviniente: {src['interviniente']}")
        print(f"  Tipo iniciativa: {src['tipo_iniciativa']}")
        print(f"  Score fusionado: {round(src['fused_score'], 6)}")
        print("-" * 60)