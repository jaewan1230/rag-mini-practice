from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

# 문서 로딩
loader = TextLoader("document.md")
docs = loader.load()

# 임베딩 및 Vector 저장
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Retriever + QA 체인
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 질문
query = "전세사기 피해자 신청 기한은 언제까지야?"
answer = qa.run(query)

print("🧠 Answer:", answer)
