from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# 1. 载入你的数据
file_path = "sample.txt"  # 你的文本数据
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

# 2. 拆分文本
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# 3. 使用 Ollama 的 Embedding 模型
embedding_model = OllamaEmbeddings(model="hf.co/Wonghehehe/model:latest")  # 适用于 Ollama 的嵌入模型

# 4. 创建 FAISS 向量数据库
vector_db = FAISS.from_documents(chunks, embedding_model)

# 5. 初始化 LLM
# Initialize the model
model = OllamaLLM(model="hf.co/Wonghehehe/model:latest")

# 6. 创建 RAG 处理链
retriever = vector_db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(model, retriever=retriever)

while True:
    # Get user input
    user_input = input("You: ")

    # 生成模型的响应
    response = qa_chain.invoke(user_input)

    # 输出模型的响应
    print(f"Model: {response}")
