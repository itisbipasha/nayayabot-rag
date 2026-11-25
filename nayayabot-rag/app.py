import os, re, json
import numpy as np, faiss, gradio as gr
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

chunks = json.load(open("chunks.json"))
vecs = np.load("vecs.npy")
texts = [c["text"] for c in chunks]

emb = SentenceTransformer("BAAI/bge-small-en-v1.5")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)
tok = lambda s: re.findall(r"\w+", s.lower())
bm25 = BM25Okapi([tok(t) for t in texts])

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-120b"
PREFIX = "Represent this sentence for searching relevant passages: "

def retrieve(query, k=20, final_k=5):
    qv = emb.encode([PREFIX + query], normalize_embeddings=True)
    dense = index.search(qv, k)[1][0].tolist()
    sparse = np.argsort(bm25.get_scores(tok(query)))[::-1][:k].tolist()
    s = {}
    for ranking in (dense, sparse):
        for r, i in enumerate(ranking):
            s[i] = s.get(i, 0) + 1 / (60 + r)
    fused = sorted(s, key=s.get, reverse=True)[:k]
    rr = reranker.predict([(query, texts[i]) for i in fused])
    return [fused[i] for i in np.argsort(rr)[::-1][:final_k]]

SYSTEM = """You answer questions about Indian law using ONLY the provided context.
Never use outside knowledge. If the answer is not clearly stated in the context,
reply exactly: "I could not find this in the provided documents."
Otherwise, cite every claim as [Act, Section X, p.Y].
Be concise. End with: "This is general information, not legal advice." """

def chat(message, history):
    ids = retrieve(message)
    ctx = "\n\n".join(
        f"[{chunks[i]['act']}, Section {chunks[i]['section']}, p.{chunks[i]['page']}]\n{chunks[i]['text']}"
        for i in ids)
    r = client.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=1500,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {message}"}])
    sources = "\n".join(
        f"- {chunks[i]['act']}, Section {chunks[i]['section']}, p.{chunks[i]['page']}" for i in ids)
    return r.choices[0].message.content + "\n\n**Retrieved sources:**\n" + sources

gr.ChatInterface(
    chat,
    title="NyayaBot: Indian Law Q&A",
    description="Cited answers over Indian statutes (RTI, Consumer Protection, and more). Not legal advice.",
    examples=["How long does a PIO have to respond to an RTI request?",
              "What is a product liability action under the Consumer Protection Act?"],
).launch()