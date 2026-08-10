import os
import warnings
import torch
from opensearchpy import OpenSearch # type: ignore
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
TOP_K = 3  # número de fragmentos a recuperar
MAX_CHUNK_TOKENS = 300  # tokens por fragmento
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore", category=UserWarning)  # suprime warnings SSL

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
# CARGAR MODELO Y TOKENIZER (4-bit)
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
    #load_in_4bit=True,            # <--- carga cuantizado a 4 bits
    #torch_dtype=torch.float16,    # usar float16 para el resto
    dtype=torch.float16,
    low_cpu_mem_usage=True
)

# ----------------------------------------
# FUNCIÓN PARA RECUPERAR CHUNKS
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
            "legislatura": src.get("legislatura"),
            "fecha": f"{src.get('dia')} {src.get('mes')} {src.get('anio')}",
            "interviniente": src.get("interviniente"),
            "tipo_iniciativa": src.get("tipo_iniciativa"),
            "score_opensearch": score
        })


    return chunks, sources


# ----------------------------------------
# FUNCIÓN DE GENERACIÓN DE RESPUESTA
# ----------------------------------------
def generate_answer(query, chunks):
    context = "\n\n".join(chunks)
    prompt = f"Usando el siguiente contexto de documentos, responde la pregunta de manera clara y precisa.\n\nContexto:\n{context}\n\nPregunta: {query}\nRespuesta:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    # recortar el prompt inicial
    answer = answer.replace(prompt, "").strip()
    return answer

# ----------------------------------------
# BUCLE INTERACTIVO
# ----------------------------------------
print("=== RAG con BM25 + Mistral 7B 4-bit ===")
while True:
    query = input("\nTú: ")
    if query.lower() in ["quit", "salir", "exit"]:
        break

    print("\nRecuperando fragmentos relevantes...")
    chunks, sources  = retrieve_chunks(query)

    if not chunks:
        print("No se encontraron fragmentos relevantes.")
        continue

    print("Generando respuesta...")
    answer = generate_answer(query, chunks)
    print(f"\nBot: {answer}")

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

