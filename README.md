# 🧠 RAG Mini Practice

Retrieval-Augmented Generation 실습 프로젝트입니다.  
LangChain과 FAISS를 이용해 간단한 문서 기반 QA 챗봇을 구현해 봅니다.

## 🔍 목표
- RAG 구성 요소에 대한 이해
- 문서 임베딩 + 벡터 검색 + LLM 응답 흐름 학습
- 향후 도메인 적용(부동산, HomeFit)을 위한 기반 마련

## 🛠 사용 기술
- Python
- LangChain
- FAISS
- OpenAI Embeddings

## 📁 구조

```plaintext
User Query
   ↓
Retriever (FAISS + Embeddings)
   ↓
Prompt Template (문서 + 질문 포함)
   ↓
LLM 응답 (OpenAI or 기타)
