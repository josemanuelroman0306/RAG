import os
import warnings
import json
import numpy as np
import torch
import re
from tqdm import tqdm
from opensearchpy import OpenSearch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ----------------------------------------
# CONFIGURACIÓN
# ----------------------------------------

OS_HOST = "localhost"
OS_PORT = 9200
OS_USER = "admin"
OS_PASS = "Tu Contraseña"
INDEX_NAME = "iniciativas_chunks2"

DATASET_FILE = "PreguntasEvalRAG2.json"

TOP_K = 3
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

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
# CARGAR DATASET
# ----------------------------------------

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"Preguntas en dataset: {len(dataset)}")

# ----------------------------------------
# CARGAR MODELO
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
# RECUPERAR CHUNKS (BM25)
# ----------------------------------------

def retrieve_chunks(query, top_k=TOP_K):

    body = {
        "size": top_k,
        "query": {
            "match": {
                "texto": {
                    "query": query
                }
            }
        }
    }

    resp = client.search(index=INDEX_NAME, body=body)

    chunks = []
    sources = []

    for hit in resp["hits"]["hits"]:

        src = hit["_source"]
        score = hit["_score"]

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

        chunks.append(bloque)

        sources.append({
            "archivo": src.get("archivo"),
            "chunk_id": src.get("chunk_id"),
            "score": score
        })

    return chunks, sources


# ----------------------------------------
# GENERAR RESPUESTA
# ----------------------------------------

def generate_answer(query, chunks):

    context = "\n\n".join(chunks)

    prompt = f"""
Responde la siguiente pregunta usando únicamente el contexto proporcionado.

Si la respuesta no está completamente en el contexto, responde parcialmente con lo que haya disponible.

No inventes información.

CONTEXTO:
{context}

PREGUNTA:
{query}

RESPUESTA:
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
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    answer = decoded[len(prompt):].strip()

    return answer


# ----------------------------------------
# EVALUAR RETRIEVAL
# ----------------------------------------

def evaluate_retrieval(sample, sources):
    relevant_chunks = sample.get("relevant_chunks", [])
    archivo = sample.get("archivo", "")

    # 1. Convertir a enteros y eliminar duplicados (ej. [3,3,4] -> {3,4})
    relevant_set = set(int(chunk) for chunk in relevant_chunks)

    # 2. Extraer los chunks recuperados que pertenecen al mismo archivo
    # Forzamos int() para asegurar que haga match con los del JSON
    retrieved_set = set(
        int(s["chunk_id"]) for s in sources 
        if s["archivo"] == archivo and s.get("chunk_id") is not None
    )

    # 3. Calcular la intersección (chunks que el RAG recuperó correctamente)
    intersection_len = len(relevant_set.intersection(retrieved_set))

    # --- MÉTRICAS ---
    
    # HIT RATE @ K: ¿Recuperó AL MENOS UN chunk relevante? (1 = Sí, 0 = No)
    hit = 1 if intersection_len > 0 else 0

    # PRECISION @ K: De los TOP_K documentos que trajimos, ¿qué porcentaje eran relevantes?
    precision = intersection_len / TOP_K

    # RECALL @ K: De TODOS los chunks que debíamos encontrar, ¿qué porcentaje trajimos?
    if len(relevant_set) > 0:
        recall = intersection_len / len(relevant_set)
    else:
        recall = 0.0

    return hit, precision, recall


# ----------------------------------------
# EVALUAR MÉTRICAS RAG
# ----------------------------------------

def evaluate_rag_metrics(query, chunks, answer):

    context = "\n\n".join(chunks)

    prompt = f"""
You are an expert evaluator of Retrieval-Augmented Generation systems. 

QUESTION: {query} 

CONTEXT: {context} 

GENERATED ANSWER: {answer} 

Rate from 1 to 5: 
1. context_relevance (Is the context useful for answering?) 
2. faithfulness (Is the answer grounded in the context?) 
3. answer_relevance (Does the answer respond to the question?) 

Return ONLY valid JSON:

{{
  "context_relevance": int,
  "faithfulness": int,
  "answer_relevance": int
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

    except Exception as e:
        print("Parse error:", response)

    return 1, 1, 1


# ----------------------------------------
# EVALUACIÓN
# ----------------------------------------

print("\n=== Evaluando RAG BM25 ===")

results = []

hit_list = []
precision_list = []
recall_list = []

context_list = []
faithfulness_list = []
answer_rel_list = []

for sample in tqdm(dataset):

    query = sample["query"]
    ground_truth = sample["ground_truth_answer"]

    chunks, sources = retrieve_chunks(query)

    if not chunks:
        continue

    answer = generate_answer(query, chunks)

    hit, precision, recall = evaluate_retrieval(sample, sources)

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
        "ground_truth": ground_truth,
        "hit": hit,
        "precision": precision,
        "recall": recall,
        "context_relevance": context_score,
        "faithfulness": faith_score,
        "answer_relevance": answer_score
    })


# ----------------------------------------
# MÉTRICAS FINALES
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

with open("resultados_rag_bm25.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nResultados guardados en resultados_rag_bm25.json")