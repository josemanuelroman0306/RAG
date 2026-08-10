import os
import warnings
import json
import numpy as np
import torch
from tqdm import tqdm
from opensearchpy import OpenSearch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

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

DATASET_FILE = "PreguntasEvalRAG.json"


TOP_K = 5
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
# CARGAR MODELO (4bit)
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
    torch_dtype=torch.float16,
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

    for hit in resp['hits']['hits']:

        src = hit['_source']
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
Usando el siguiente contexto de documentos, responde la pregunta de forma clara.

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

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = answer.replace(prompt, "").strip()

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
# EVALUACIÓN
# ----------------------------------------
print("\n=== Evaluando RAG BM25 ===")

results = []

hit_list = []
precision_list = []
recall_list = []

for sample in tqdm(dataset):

    query = sample["query"]
    ground_truth = sample["ground_truth_answer"]

    chunks, sources = retrieve_chunks(query)

    if not chunks:
        continue

    #answer = generate_answer(query, chunks)

    hit, precision, recall = evaluate_retrieval(sample, sources)

    hit_list.append(hit)
    precision_list.append(precision)
    recall_list.append(recall)

    results.append({
        "query": query,
        #"generated_answer": answer,
        "ground_truth": ground_truth,
        "hit": hit,
        "precision": precision,
        "recall": recall
    })

# ----------------------------------------
# MÉTRICAS FINALES
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
with open("resultados_rag_bm25R.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nResultados guardados en resultados_rag_bm25R.json")