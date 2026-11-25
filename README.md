
# NyayaBot: RAG Assistant for Indian Laws

Ask questions about Indian laws and get answers with section and page citations.

## What it does
- Reads Indian acts (RTI Act 2005, Consumer Protection Act 2019, and more)
- Splits them into section-aware chunks
- Finds relevant text using BM25 + dense search, merged with Reciprocal Rank Fusion
- Re-ranks results with a cross-encoder
- Answers with gpt-oss-120b (via Groq), only from the retrieved text
- Says "I could not find this in the provided documents" when the answer isn't there

## Architecture
PDF -> section-aware chunks -> BM25 + dense search -> RRF -> cross-encoder rerank -> LLM -> cited answer

## Results
Evaluated on 50 hand-verified questions (Hit@5 = correct act and page within top 5, plus or minus 1 page).

| Method | Hit@5 | MRR |
|---|---|---|
| BM25 | 0.94 | 0.837 |
| Dense (bge-small) | 0.92 | 0.811 |
| Hybrid (RRF) | 0.96 | 0.864 |
| Hybrid + Reranker | 1.00 | 0.920 |

## Tech stack
Python, PyMuPDF, sentence-transformers (bge-small), FAISS, rank_bm25, cross-encoder, Groq API, Gradio

## How to run locally
1. Install Python 3.10 to 3.12
2. `pip install -r requirements.txt`
3. Get a free API key from console.groq.com
4. Set the key:
   - Windows: `set GROQ_API_KEY=your_key`
   - Mac/Linux: `export GROQ_API_KEY=your_key`
5. Run `python app.py` and open http://127.0.0.1:7860

## Limitations
- Questions were LLM-generated from the source text and manually filtered, so real-world performance is likely lower than reported.
- Matching is page-based. English only. Scanned PDFs are not supported.
- Not legal advice.

## Author
Your Name | your LinkedIn link
