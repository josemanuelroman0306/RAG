import os
import json
import numpy as np
from tqdm import tqdm
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
import torch
import warnings
import urllib3
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

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
# EMBEDDINGS
# ----------------------------------------

print(f"Usando dispositivo: {DEVICE}")

embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

def embed_query(text):
    text = "query: " + text
    vec = embedder.encode(text, normalize_embeddings=True)
    return vec.tolist()

# ----------------------------------------
# MODELO GENERATIVO
# ----------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

max_memory = {
    0: "6GB",   # GPU
    "cpu": "32GB"
}

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    max_memory=max_memory,
    quantization_config=bnb_config,
    dtype=torch.float16,
    low_cpu_mem_usage=True
)

# ----------------------------------------
# BM25
# ----------------------------------------

def retrieve_bm25(query, size=20):

    body = {
        "size": size,
        "query": {
            "match": {
                "texto": query
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
# RRF
# ----------------------------------------

def reciprocal_rank_fusion(bm25_hits, knn_hits):

    scores = {}
    doc_info = {}
    doc_text = {}

    for hit in bm25_hits + knn_hits:

        doc_id = hit["_id"]
        src = hit["_source"]

        doc_info[doc_id] = {
            "archivo": src.get("archivo"),
            "chunk_id": src.get("chunk_id")
        }

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

        doc_text[doc_id] = bloque

    # BM25 scores
    for rank, hit in enumerate(bm25_hits):
        doc_id = hit["_id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (RRF_K + rank + 1)

    # KNN scores
    for rank, hit in enumerate(knn_hits):
        doc_id = hit["_id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (RRF_K + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    retrieved = []
    chunks = []

    for doc_id, score in ranked[:TOP_K]:

        info = doc_info[doc_id]

        retrieved.append({
            "archivo": info["archivo"],
            "chunk_id": info["chunk_id"]
        })

        chunks.append(doc_text[doc_id])

    return retrieved, chunks

# ----------------------------------------
# GENERAR RESPUESTA
# ----------------------------------------

def generate_answer(query, chunks):

    context = "\n\n".join(chunks)

    prompt = f"""
Usando el siguiente contexto parlamentario responde la pregunta.

Contexto:
{context}

Pregunta: {query}

Respuesta:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(DEVICE)

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    answer = decoded.replace(prompt, "").strip()

    return answer

# ----------------------------------------
# MÉTRICAS RAG
# ----------------------------------------

def evaluate_rag_metrics(query, chunks, answer):

    context = "\n\n".join(chunks)

    prompt = f"""
You are an expert evaluator of Retrieval-Augmented Generation systems.

QUESTION:
{query}

CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Rate from 1 to 5:

1. context_relevance
2. faithfulness
3. answer_relevance

Return ONLY valid JSON:

{{
"context_relevance": number,
"faithfulness": number,
"answer_relevance": number
}}
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(DEVICE)

    output = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        #temperature=0.0,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    response = decoded[len(prompt):].strip()

    try:

        response = response.replace("```json", "").replace("```", "")

        match = re.search(r"\{.*?\}", response, re.DOTALL)

        if match:

            metrics = json.loads(match.group())

            context_score = int(metrics.get("context_relevance", 1))
            faith_score = int(metrics.get("faithfulness", 1))
            answer_score = int(metrics.get("answer_relevance", 1))

            context_score = max(1, min(context_score, 5))
            faith_score = max(1, min(faith_score, 5))
            answer_score = max(1, min(answer_score, 5))

            return context_score, faith_score, answer_score

    except:
        pass

    return 1,1,1

# ----------------------------------------
# DATASET
# ----------------------------------------

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"\nPreguntas en dataset: {len(dataset)}")

# ----------------------------------------
# EVALUACIÓN
# ----------------------------------------

results = []

hit_list = []
precision_list = []
recall_list = []

context_list = []
faithfulness_list = []
answer_rel_list = []

print("\n=== Evaluando RAG Hybrid RRF ===\n")

for sample in tqdm(dataset):

    query = sample["query"]
    relevant_chunks = sample["relevant_chunks"]
    archivo = sample["archivo"]

    bm25_hits = retrieve_bm25(query)
    knn_hits = retrieve_knn(query)

    retrieved, chunks = reciprocal_rank_fusion(bm25_hits, knn_hits)

    answer = generate_answer(query, chunks)

    # --- REEMPLAZA ESTE BLOQUE EN TU SCRIPT RRF ---

    # 1. Convertimos a enteros para asegurar el match
    relevant_set = set(int(chunk) for chunk in relevant_chunks)
    
    retrieved_ids = [
        int(r["chunk_id"]) for r in retrieved
        if r["archivo"] == archivo and r.get("chunk_id") is not None
    ]
    retrieved_set = set(retrieved_ids)

    # 2. Intersección segura
    intersection = relevant_set.intersection(retrieved_set)
    intersection_len = len(intersection)

    # 3. Métricas
    hit = 1 if intersection_len > 0 else 0
    precision = intersection_len / TOP_K
    recall = intersection_len / len(relevant_set) if len(relevant_set) > 0 else 0.0

    context_score, faith_score, answer_score = evaluate_rag_metrics(
        query,
        chunks,
        answer
    )

    hit_list.append(hit)
    precision_list.append(precision)
    recall_list.append(recall)

    context_list.append(context_score)
    faithfulness_list.append(faith_score)
    answer_rel_list.append(answer_score)

    results.append({
        "query": query,
        "generated_answer": answer,
        "hit": hit,
        "precision": precision,
        "recall": recall,
        "context_relevance": context_score,
        "faithfulness": faith_score,
        "answer_relevance": answer_score
    })

# ----------------------------------------
# RESULTADOS
# ----------------------------------------

hit_rate = np.mean(hit_list)
precision_k = np.mean(precision_list)
recall_k = np.mean(recall_list)

context_avg = np.mean(context_list)
faithfulness_avg = np.mean(faithfulness_list)
answer_rel_avg = np.mean(answer_rel_list)

print("\n===== RESULTADOS RETRIEVAL =====\n")

print(f"HitRate@{TOP_K}: {hit_rate:.3f}")
print(f"Precision@{TOP_K}: {precision_k:.3f}")
print(f"Recall@{TOP_K}: {recall_k:.3f}")

print("\n===== MÉTRICAS RAG =====\n")

print(f"Context Relevance: {context_avg:.2f} / 5")
print(f"Faithfulness: {faithfulness_avg:.2f} / 5")
print(f"Answer Relevance: {answer_rel_avg:.2f} / 5")

# ----------------------------------------
# GUARDAR RESULTADOS
# ----------------------------------------

summary = {
    "HitRate@3": float(hit_rate),
    "Precision@3": float(precision_k),
    "Recall@3": float(recall_k),
    "ContextRelevance": float(context_avg),
    "Faithfulness": float(faithfulness_avg),
    "AnswerRelevance": float(answer_rel_avg)
}

output = {
    "summary": summary,
    "samples": results
}

with open("resultados_rag_rrf.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print("\nResultados guardados en resultados_rag_rrf.json")