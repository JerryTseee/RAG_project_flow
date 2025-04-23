# using Hugging face to achieve RAG
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import faiss
import os
from transformers import pipeline

embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def get_embedding(text):
    return embedding_model.encode(text)

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

query = "Jerry is handsome"
documents = [
    "Jerry's face is pretty",
    "Jerry is the most handsome man at HKU",
    "I am idiot"
]

query_vector = get_embedding(query)
documents_vectors = [get_embedding(i) for i in documents]
for i, doc_vec in enumerate(documents_vectors):
    print(f"\nDocument {i+1}: {documents[i]}")
    print(f"Euclidean distance: {euclidean_distance(query_vector, doc_vec):.4f}")
    print(f"Cosine similarity: {cosine_similarity(query_vector, doc_vec):.4f}")

print(" ")
# load the documents and separate them into chunks
"""
how to chunk?
1. split by sentence
2. split by string number
3. split by a sliding window
4. split by recursion method
"""
text = "My name is TSE WANG POK (Jerry), I am very handsome, I like NLP"

# 1
sentences = re.split(r"(?<=[(),.!?])\s+", text)
for i, chunk in enumerate(sentences):
    print(f"chunk {i+1}: {chunk}")
print(" ")
# 2
def split_by_fixed_char_count(text, count):
    return [text[i:i+count] for i in range(0, len(text), count)]
chunks = split_by_fixed_char_count(text, 15)
for i, chunk in enumerate(chunks):
    print(f"chunk {i+1}: {chunk}")
print(" ")
# 3
def sliding_window_chunks(text, window_size, stride):
    return [text[i:i+window_size] for i in range(0, len(text), stride)]
chunks = sliding_window_chunks(text, 10, 5)
for i, chunk in enumerate(chunks):
    print(f"chunk {i+1}: {chunk}")
print(" ")
# 4
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 20,
    chunk_overlap = 5,
    length_function = len,
)
chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"chunk {i+1}: {chunk}")
print(" ")