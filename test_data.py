from data_loader import load_all
from chunker import chunk_all

docs = load_all()
print(f'Loaded {len(docs)} documents')

print('Sample docs:')
for i, doc in enumerate(docs[:3]):
    print(f'Doc {i}: {doc["id"]} - {doc["source"]} - {len(doc["content"])} chars')
    print(f'Content preview: {doc["content"][:200]}...')
    print()

chunks = chunk_all(docs)
print(f'Created {len(chunks)} chunks')

print('Sample chunks:')
for i, chunk in enumerate(chunks[:3]):
    print(f'Chunk {i}: {chunk["id"]} - {chunk["source"]} - {len(chunk["text"])} chars')
    print(f'Text preview: {chunk["text"][:200]}...')
    print()