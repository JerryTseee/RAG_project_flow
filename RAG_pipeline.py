from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import faiss
import os
from transformers import pipeline

# 1. 初始化模型
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
generator = pipeline('text-generation', model='distilgpt2')  # 小型生成模型

# 2. 文档加载和预处理
def load_and_chunk_documents(file_path):
    # 加载文档
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # 分块处理
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=30,
        chunk_overlap=10,
        length_function=len
    )
    chunks = splitter.split_documents(documents)
    return [chunk.page_content for chunk in chunks]

# 3. 向量化存储
def create_vector_store(texts):
    # 生成嵌入向量
    embeddings = embedding_model.encode(texts)
    
    # 创建FAISS索引 （用来高效相似度搜索优化的数据结构，可以快速找到与查询向量最相似的向量集合）
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings, texts

# 4. 检索相关文档
def retrieve_documents(query, index, texts, k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    
    retrieved_docs = []
    for idx in indices[0]:
        retrieved_docs.append(texts[idx])
    return retrieved_docs

# 5. 生成回答
def generate_response(query, context):
    prompt = f"""基于以下上下文信息回答问题。如果无法从上下文中得到答案，请回答"我不知道"。
    
    上下文: {context}
    
    问题: {query}
    回答:"""
    
    response = generator(
        prompt,
        max_new_tokens=150,
        num_return_sequences=1,
        temperature=0.7,
        truncation=True
    )
    return response[0]['generated_text']

# 6. 完整RAG流程
def run_rag_system(query, file_path):
    # 加载和分块文档
    chunks = load_and_chunk_documents(file_path)
    
    # 创建向量存储
    index, embeddings, texts = create_vector_store(chunks)
    
    # 检索相关文档
    retrieved_docs = retrieve_documents(query, index, texts)
    context = "\n\n".join(retrieved_docs)
    
    # 生成回答
    response = generate_response(query, context)
    
    return response, retrieved_docs

# 7. 测试RAG系统
if __name__ == "__main__":
    # 创建测试文档
    test_doc = """Jerry TSE是香港大学计算机科学系的教授。
    他专长于自然语言处理和机器学习。
    Jerry在2020年获得了ACM杰出科学家奖。
    他开发了多个开源NLP工具包。
    Jerry的办公室位于HKU的CS大楼8层。"""
    
    with open("test_doc.txt", "w", encoding="utf-8") as f:
        f.write(test_doc)
    
    # 运行查询
    query = "Jerry TSE获得了什么奖项？"
    response, sources = run_rag_system(query, "test_doc.txt")
    
    print("\n=== 用户问题 ===")
    print(query)
    
    print("\n=== 检索到的相关文档 ===")
    for i, doc in enumerate(sources, 1):
        print(f"[文档{i}]: {doc}")
    
    print("\n=== 生成的回答 ===")
    print(response.split("回答:")[1].strip())
