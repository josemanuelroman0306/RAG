import os
import warnings
import torch
from opensearchpy import OpenSearch  # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
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
INDEX_NAME = "iniciativas_chunks2_emb"   # mismo índice (texto + embedding)
TOP_K = 3

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
    verify_certs=False,
    timeout=60,
    max_retries=5,
    retry_on_timeout=True
)

# ----------------------------------------
# MODELO GENERATIVO (Mistral 7B 4-bit)
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
# MODELO DE EMBEDDINGS (E5)
# ----------------------------------------
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

def embed_query(text: str):
    text = "query: " + text   # obligatorio para E5
    vec = embedder.encode(text, normalize_embeddings=True)
    return vec.tolist()

# ----------------------------------------
# RECUPERACIÓN HÍBRIDA (BM25 + kNN)
# ----------------------------------------
def retrieve_chunks_hybrid(query, top_k=TOP_K):
    query_vector = embed_query(query)

    body = {
        "size": top_k,
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "match": {
                            "texto": {
                                "query": query
                            }
                        }
                    },
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_vector,
                                "k": top_k
                            }
                        }
                    }
                ]
            }
        }
    }

    resp = client.search(index=INDEX_NAME, body=body)

    chunks = []
    sources =[]

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
            "legislatura": src.get("legislatura"),
            "fecha": f"{src.get('dia')} {src.get('mes')} {src.get('anio')}",
            "interviniente": src.get("interviniente"),
            "tipo_iniciativa": src.get("tipo_iniciativa"),
            "score_opensearch": score
        })

    return chunks, sources

# ----------------------------------------
# GENERACIÓN DE RESPUESTA
# ----------------------------------------
def generate_answer(query, chunks):
    context = "\n\n".join(chunks)

    prompt = f"""Usando el siguiente contexto de documentos parlamentarios, responde la pregunta de forma clara, precisa y basada únicamente en el contexto proporcionado.

Contexto:
{context}

Pregunta: {query}

Respuesta:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(DEVICE)

    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = answer.replace(prompt, "").strip()

    return answer

# ----------------------------------------
# BUCLE INTERACTIVO
# ----------------------------------------
print("=== RAG HÍBRIDO (BM25 + kNN) + Mistral 7B 4-bit ===")

while True:
    query = input("\nTú: ")

    if query.lower() in ["quit", "salir", "exit"]:
        break

    print("\nRecuperando fragmentos con búsqueda híbrida...")
    
    try:
        chunks, sources = retrieve_chunks_hybrid(query)
    except Exception as e:
        print("Error en consulta híbrida:")
        print(e)
        break

    if not chunks:
        print("No se encontraron fragmentos relevantes.")
        continue

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
        print(f"  Score OpenSearch: {round(src['score_opensearch'], 6)}")
        print("-" * 60)