from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

# 문서 로딩
loader = TextLoader("document.md", encoding='utf-8')
docs = loader.load()

# 임베딩 및 Vector 저장
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Retriever + QA 체인
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 질문
query = "전세사기 피해자 신청 기한은 언제까지야?"
answer = qa.run(query)

print("🧠 Answer:", answer)
