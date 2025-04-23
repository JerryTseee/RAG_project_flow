from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key="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")

# from text to word embeddings
def get_embedding(text, model="text-embedding-3-large"):
    data = client.embeddings.create(input = text, model = model).data
    return [x.embedding for x in data]

test_query = [{"What is the capital of France?"}]
vec = get_embedding(test_query)
print("word vector: " + str(vec))
print("vector length: " + str(len(vec))) 


# after we get the word embeddings, we can calculate the similarity using Euclidian distance
import numpy as np
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

query = "Jerry is handsome"
documents = ["Jerry's face is pretty",
             "Jerry is the most handsome man at HKU",
             "I am idiot"
             ]
query_vector = get_embedding([query])[0]
documents_vectors = [get_embedding(i) for i in documents]
count = 0
for i in documents_vectors:
    print(f"No.{count}")
    print("Euclidean distance: " + str(euclidean_distance(query_vector, i[0])))
    print("Cosine similarity: " + str(cosine_similarity(query_vector, i[0])))
    count += 1


# load the documents and separate them into chunks
"""
how to chunk?
1. split by sentence
2. split by string number
3. split by a sliding window
4. split by recursion method
"""

# split by sentence
import re
text = "My name is TSE WANG POK (Jerry), I am very handsome, I like NLP"
sentences = re.split(r'(.|?|!\...\...)', text) # 结束标点符号
chunks = [sentence + (punctuation if punctuation else '') for sentence, punctuation in zip(sentences[::2], sentences[1::2])]
for i, chunk in enumerate(chunks):
    print(f"Chunk No.{i+1}: {len(chunk)}: {chunk}")



# split by string number
text = "My name is TSE WANG POK (Jerry), I am very handsome, I like NLP"

def split_by_fixed_char_count(text, count):
    return [text[i:i+count] for i in range(0, len(text), count)]

chunks = split_by_fixed_char_count(text, 100)
for i, chunk in enumerate(chunks):
    print(f"Chunk No.{i+1}: {len(chunk)}: {chunk}")



# split by a sliding window
text = "My name is TSE WANG POK (Jerry), I am very handsome, I like NLP"

def sliding_window_chunks(text, chunk_size, stride):
    return [text[i:i+chunk_size] for i in range(0, len(text), stride)]

chunks = sliding_window_chunks(text, 4, 2)
for i, chunk in enumerate(chunks):
    print(f"Chunk No.{i+1}: {len(chunk)}: {chunk}")



# split by recursion method (need to use langchain)
text = "My name is TSE WANG POK (Jerry), I am very handsome, I like NLP"

from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 50,
    chunk_overlap = 10,
    length_function = len,
)

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk No.{i+1}: {len(chunk)}: {chunk}")




# Keywords search
"""
how to conduct keyword search?
1. use keyword to match the data
2. use keyword and its meaning to match the data
"""