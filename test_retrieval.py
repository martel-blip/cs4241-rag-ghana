from data_loader import load_all
from chunker import chunk_all
from modules import Embedder, NumpyVectorStore, HybridRetriever

# Load and chunk data
docs = load_all()
chunks = chunk_all(docs)
print(f"Loaded {len(docs)} docs, created {len(chunks)} chunks")

# Test embedding and retrieval
embedder = Embedder()
vector_store = NumpyVectorStore()

print("Building vector store...")
embeddings = embedder.encode([chunk["text"] for chunk in chunks])
vector_store.build(embeddings, chunks)

retriever = HybridRetriever(
    embedder=embedder,
    vector_store=vector_store,
    chunks=chunks,
    k_dense=15,
    k_bm25=15,
    k_final=5,
    dense_weight=0.6,
    bm25_weight=0.4
)

# Test query
query = "Who won the 2020 presidential election in Ghana?"
print(f"\nTesting query: {query}")

hits = retriever.retrieve(query, top_k=5)
print(f"Retrieved {len(hits)} results:")

for i, hit in enumerate(hits):
    print(f"\n{i+1}. Score: {hit.score:.3f}")
    print(f"   Source: {hit.source}")
    print(f"   Text: {hit.text[:200]}...")