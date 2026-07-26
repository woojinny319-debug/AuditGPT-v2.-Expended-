# AuditGPT

K-IFRS 기준서 · DART 공시 · 회계법인 KAM/가이드북을 근거로 답변하는 다산업 회계감사 RAG 챗봇입니다.

## 무엇을 하는가

- **K-IFRS 11개 기준서**(수익, 리스, 금융상품, 유형자산, 무형자산 등) 기반 회계 원칙 질의응답
- **DART 공시 데이터**: 5개 산업(제약·바이오, 건설, 에너지·유틸리티, 외식, 소비재·유통) 총 79개 상장사의 최근 사업보고서
- **삼일회계법인 산업별 가이드북/KAM** 문서 기반 실무 사례
- 질문에 언급된 기업·산업을 자동 감지해 관련 문서만 우선 검색하고, 답변마다 근거 문서를 `[S#]` 형태로 인용

## 기술 스택

- **검색**: BM25(키워드) + Semantic(임베딩) 하이브리드 검색 → FlashRank 재순위화(2-stage)
- **벡터 DB**: ChromaDB (로컬 persistent)
- **LLM/임베딩**: OpenAI (`gpt-4o`, `text-embedding-3-small`)
- **백엔드**: FastAPI
- **프론트엔드**: 정적 HTML/CSS/JS (빌드 도구 없음)

## 프로젝트 구조

```
industries.py       # 산업별 대상 기업·가이드북·DART 수집 설정 (단일 진실 공급원)
rag_engine.py        # 리트리버(하이브리드 검색 + FlashRank) 구성
prompts.py            # 시스템 프롬프트, 인용 포맷
dart_ingest.py        # DART 공시 수집 스크립트
backend/main.py       # FastAPI 앱 (/api/chat, /api/health)
frontend/              # 정적 웹 UI
deploy/                 # systemd/nginx 배포 설정 예시
```

## 실행 방법

```bash
pip install -r requirements.txt
```

`.env` 파일에 API 키 설정:

```
OPENAI_API_KEY=...
DART_API_KEY=...
```

DART 데이터 수집(최초 1회, 산업별로 나눠서 실행 가능):

```bash
python dart_ingest.py --industry bio
```

서버 실행:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

브라우저에서 `http://127.0.0.1:8000` 접속.
