"""FastAPI backend for the AuditGPT web chatbot.

Streamlit(app.py)의 build_answer()/retrieve_parallel()/_build_source_context() 로직을
그대로 이식한 것 — 새 로직이 아니라 UI 계층만 교체(AuditGPT_웹전환_구현계획서 3-2절 기준).
prompts.py는 손대지 않고 그대로 import해서 쓴다.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

# rag_engine.py / prompts.py가 프로젝트 루트에 있으므로 루트를 sys.path에 추가한다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

import industries
from prompts import SYSTEM_PROMPT, build_user_prompt, format_cited_docs
from rag_engine import get_dart_retriever, get_guidebook_retriever, get_kifrs_retriever


class _State:
    """app.py의 st.cache_resource 캐시를 대체하는 프로세스 전역 싱글턴 보관소."""

    embeddings: OpenAIEmbeddings | None = None
    llm: ChatOpenAI | None = None
    kifrs_retriever: BaseRetriever | None = None
    guidebook_retriever: BaseRetriever | None = None
    dart_retriever: BaseRetriever | None = None


_state = _State()


def _load_retrievers() -> None:
    """앱 시작 시 1회만 리트리버를 로드한다(요청마다 재로딩 방지)."""
    _state.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    _state.llm = ChatOpenAI(model="gpt-4o", temperature=0)
    _state.kifrs_retriever = get_kifrs_retriever(_state.embeddings)
    _state.guidebook_retriever = get_guidebook_retriever(_state.embeddings)
    _state.dart_retriever = get_dart_retriever(_state.embeddings)  # 데이터 없으면 None


app = FastAPI(title="AuditGPT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _on_startup() -> None:
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key or openai_key == "your_openai_api_key_here":
        raise RuntimeError("OPENAI_API_KEY가 없습니다. .env를 확인해 주세요.")
    _load_retrievers()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] = []


class SourceItem(BaseModel):
    sid: str
    source: str
    company: str
    industry: str = ""
    section: str
    page: str
    url: str
    content: str
    category: str = ""  # "kifrs" | "dart" | "guidebook" — 프론트에서 출처 그룹핑에 사용


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


def retrieve_parallel(query: str) -> tuple[list[Document], list[Document], list[Document]]:
    # [중요] app.py와 동일하게 순차 검색 유지. chromadb 1.5.x는 단일 클라이언트의 동시(멀티스레드)
    # 쿼리에서 "Nothing found on disk"(HNSW 세그먼트 레이스)를 던진다.
    kifrs_docs = _state.kifrs_retriever.invoke(query)
    guidebook_docs = _state.guidebook_retriever.invoke(query)

    # DART는 "특정 기업의 실제 공시 수치"를 다루는 소스라, 질의에 기업명이 없는 일반
    # 기준서/원칙 질문에서는 검색 자체를 건너뛴다. 이전엔 기업명이 없으면 DynamicDartRetriever가
    # 전역(전체 79개사) 검색을 했는데, 그 결과가 FlashRank 재랭킹을 거쳐도 "질문과 의미상 살짝
    # 관련 있어 보이는 아무 기업"으로 채워져 K-IFRS/가이드북보다 우선 인용되는 문제가 있었다.
    # 우선순위 1) K-IFRS, 2) 가이드북·KAM, 3) DART(기업명이 실제로 언급된 경우만)를
    # 프롬프트가 아니라 "애초에 후보에 넣지 않는" 방식으로 강제한다.
    _, positive_companies, _ = industries.detect_industries_and_companies(query)
    if positive_companies and _state.dart_retriever is not None:
        dart_docs = _state.dart_retriever.invoke(query)
    else:
        dart_docs = []
    return kifrs_docs, dart_docs, guidebook_docs


def _trim_docs(docs: list[Document], max_docs: int = 3, max_chars: int = 1200) -> list[Document]:
    trimmed: list[Document] = []
    for doc in docs[:max_docs]:
        text = (doc.page_content or "").strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...(truncated)"
        trimmed.append(Document(page_content=text, metadata=doc.metadata))
    return trimmed


def _build_source_context(
    kifrs_docs: list[Document], dart_docs: list[Document], guidebook_docs: list[Document]
) -> tuple[str, list[dict[str, str]]]:
    kifrs_docs = _trim_docs(kifrs_docs, max_docs=4, max_chars=2000)
    dart_docs = _trim_docs(dart_docs, max_docs=15, max_chars=3500)
    guidebook_docs = _trim_docs(guidebook_docs, max_docs=3, max_chars=1500)

    p1, c1 = format_cited_docs(kifrs_docs, 1)
    p2, c2 = format_cited_docs(dart_docs, len(c1) + 1)
    p3, c3 = format_cited_docs(guidebook_docs, len(c1) + len(c2) + 1)
    # prompts.py는 수정하지 않으므로, format_cited_docs가 돌려준 dict에 카테고리를
    # 여기(호출부)에서 덧붙인다 — 어느 리스트에서 왔는지는 이 시점에 이미 알고 있다.
    for item in c1:
        item["category"] = "kifrs"
    for item in c2:
        item["category"] = "dart"
    for item in c3:
        item["category"] = "guidebook"
    context = "\n\n".join([x for x in [p1, p2, p3] if x])
    return context, c1 + c2 + c3


_COMPARISON_KEYWORDS = ("비교", "차이", "대비", " vs ", "vs.", "어느 쪽", "어느쪽")


def _augment_query_for_comparison(query: str) -> str:
    """'비교해줘' 류 질문엔 마크다운 표로 정리하라는 힌트를 이번 턴의 질문 텍스트에 덧붙인다.

    prompts.py(SYSTEM_PROMPT)는 건드리지 않는다 — 시스템 프롬프트에 규칙을 추가하는 대신,
    build_user_prompt()에 넘기는 "질문" 문자열 자체에 지시를 실어 보내는 방식(호출부만 수정).
    """
    if any(kw in query for kw in _COMPARISON_KEYWORDS):
        return (
            query
            + "\n\n(참고: 위 질문은 여러 대상을 비교해달라는 요청입니다. "
            "가능하면 비교 항목을 마크다운 표(| 헤더 | ... |)로 정리해서 답변해 주세요.)"
        )
    return query


_COMPANY_BOLD_PATTERN: re.Pattern | None = None


def _get_company_bold_pattern() -> re.Pattern:
    global _COMPANY_BOLD_PATTERN
    if _COMPANY_BOLD_PATTERN is None:
        names = sorted(
            set(industries.COMPANY_TO_INDUSTRY.keys()) | set(industries.COMPANY_ALIAS_MAP.keys()),
            key=len,
            reverse=True,
        )
        pattern = r"(?<!\*\*)(" + "|".join(re.escape(n) for n in names) + r")(?!\*\*)"
        _COMPANY_BOLD_PATTERN = re.compile(pattern)
    return _COMPANY_BOLD_PATTERN


def _bold_company_names(text: str) -> str:
    """답변에 사례로 등장하는 기업명을 자동으로 **굵게** 표시한다.

    prompts.py의 서식 규칙(핵심 용어 굵게 표시 등)을 건드리는 대신, 생성이 끝난 답변
    텍스트를 후처리한다 — industries.py의 79개사 전체 명단+별칭 기준으로 매칭한다.
    """
    return _get_company_bold_pattern().sub(lambda m: f"**{m.group(0)}**", text)


def build_answer(query: str) -> tuple[str, list[dict[str, str]]]:
    kifrs_docs, dart_docs, guidebook_docs = retrieve_parallel(query)
    source_context, catalog = _build_source_context(kifrs_docs, dart_docs, guidebook_docs)
    prompt = build_user_prompt(_augment_query_for_comparison(query), source_context)
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]

    response = None
    for attempt in range(3):
        try:
            response = _state.llm.invoke(messages)
            break
        except Exception as e:
            err = str(e)
            is_tpm = ("rate_limit_exceeded" in err) or ("tokens per min" in err) or ("Error code: 429" in err)
            if not is_tpm or attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    if response is None:
        raise RuntimeError("LLM 응답 생성 실패")
    return _bold_company_names(str(response.content)), catalog


@app.get("/api/health")
def health() -> dict:
    return {
        "kifrs_loaded": _state.kifrs_retriever is not None,
        "guidebook_loaded": _state.guidebook_retriever is not None,
        "dart_loaded": _state.dart_retriever is not None,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query가 비어 있습니다.")
    try:
        answer, catalog = build_answer(req.query)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"답변 생성 중 오류: {e}") from e

    sources = [
        SourceItem(
            sid=item.get("sid", ""),
            source=item.get("source", ""),
            company=item.get("company", ""),
            industry=item.get("industry", ""),
            section=item.get("section", ""),
            page=item.get("page", ""),
            url=item.get("url", ""),
            content=item.get("content", ""),
            category=item.get("category", ""),
        )
        for item in catalog
    ]
    return ChatResponse(answer=answer, sources=sources)


# 정적 프론트엔드(index.html/style.css/app.js) 서빙. 운영 환경에서는 Nginx가 이 역할을
# 대신 맡을 수 있지만(AuditGPT_웹전환_구현계획서 3-4절), 로컬 개발/단일 프로세스 배포에서는
# FastAPI가 직접 서빙하면 별도 웹서버 없이도 바로 동작한다.
_FRONTEND_DIR = PROJECT_ROOT / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
