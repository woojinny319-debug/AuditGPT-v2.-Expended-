"""Retrieval engine with duplicate-safe ingest and 2-stage retrieval."""

from __future__ import annotations

import hashlib
import pickle
import re
import shutil
import time
from pathlib import Path
from typing import Callable

import chromadb
import fitz
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import industries

try:
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
except Exception:
    try:
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
    except Exception:
        try:
            # langchain>=1.3.x: ContextualCompressionRetriever가 langchain_classic으로 이전됨
            # (기존 두 경로 모두 여기서 실패해 FlashRank가 조용히 비활성화되고 있었음 — 버그 수정).
            from langchain_classic.retrievers import ContextualCompressionRetriever
            from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
        except Exception:
            ContextualCompressionRetriever = None
            FlashrankRerank = None

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

import os

PROJECT_ROOT = Path(__file__).parent
# [근본 원인] 프로젝트가 한글 경로("바탕 화면") 아래에 있는데, chromadb 1.5.x의 Rust HNSW
# 세그먼트 로더가 비ASCII 절대경로를 못 열어 "Error loading hnsw index"가 발생했다.
# (상대경로/ASCII 경로는 정상). 따라서 벡터 DB는 ASCII 절대경로(사용자 홈 하위)에 둔다.
# 2026-06-19 디버깅 참고.
_ASCII_DATA_HOME = os.path.join(os.path.expanduser("~"), "rag_chroma_data")
CHROMA_DIR = os.path.join(_ASCII_DATA_HOME, "chroma_db_v2")
if not CHROMA_DIR.isascii():
    # 사용자명마저 비ASCII인 극단적 경우의 안전장치: 프로젝트 상대경로로 폴백
    CHROMA_DIR = str(PROJECT_ROOT / "chroma_db_v2")
EMBED_BATCH_SIZE = 30
EMBED_BATCH_DELAY = 1.6

_SHARED_CLIENT = None


def _get_client():
    """프로세스 전체가 공유하는 단일 chromadb 클라이언트.

    [중요] 한 프로세스에서 PersistentClient나 langchain Chroma가 각자 별도의 클라이언트를
    만들면, chromadb 1.5.x에서 compactor의 backfill 충돌이 일어나 HNSW 인덱스 로드가
    비결정적으로 실패한다(2026-06-19 디버깅). 모든 chroma 접근이 이 단일 클라이언트를
    공유하도록 강제하면 충돌이 사라진다.
    """
    global _SHARED_CLIENT
    if _SHARED_CLIENT is None:
        _SHARED_CLIENT = chromadb.PersistentClient(path=CHROMA_DIR)
    return _SHARED_CLIENT


def _reset_client() -> None:
    """공유 클라이언트를 폐기해 다음 _get_client()가 *진짜로* 새 클라이언트를 만들도록 한다.

    chromadb는 PersistentClient를 path별로 내부 캐싱(SharedSystemClient)하므로, 파이썬 참조만
    None으로 두면 같은 stale 시스템을 다시 돌려준다. clear_system_cache()로 내부 캐시까지
    비워야 세그먼트 리더가 새로 초기화되어 'Nothing found on disk'가 복구된다(2026-06-20 디버깅)."""
    global _SHARED_CLIENT
    _SHARED_CLIENT = None
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception:
        try:
            from chromadb.api.client import SharedSystemClient as _SSC
            _SSC.clear_system_cache()
        except Exception:
            pass

def _load_pdf(
    pdf_path: Path,
    source_id: str,
    industry: str = "공통",
    doc_type: str = "standard",
) -> list[Document]:
    doc = fitz.open(str(pdf_path))
    out: list[Document] = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        if text.strip():
            out.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source_id,
                        "page": str(page_num + 1),
                        "file": pdf_path.name,
                        "source_url": "",
                        "industry": industry,
                        "doc_type": doc_type,
                    },
                )
            )
    doc.close()
    return out


def _chunk(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=40,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)


def _stable_doc_id(collection_name: str, doc: Document, idx: int) -> str:
    m = doc.metadata or {}
    company = str(m.get("company", ""))
    year = str(m.get("year", ""))
    section = str(m.get("section", ""))
    if company and year and section:
        return f"{company}_{year}_{section}_{idx}"
    body = f"{collection_name}|{idx}|{doc.page_content}"
    return f"{collection_name}_{hashlib.sha1(body.encode('utf-8')).hexdigest()[:16]}"


def save_to_chroma(
    collection_name: str,
    embeddings: OpenAIEmbeddings,
    docs: list[Document],
    id_builder: Callable[[Document, int], str] | None = None,
) -> None:
    if not docs:
        return
    store = Chroma(collection_name=collection_name, embedding_function=embeddings, client=_get_client())
    ids = [(id_builder(d, i) if id_builder else _stable_doc_id(collection_name, d, i)) for i, d in enumerate(docs)]
    try:
        store.delete(ids=ids)
    except Exception:
        pass
    try:
        store.add_documents(docs, ids=ids)
    except Exception as e:
        msg = str(e).lower()
        if "error loading hnsw index" in msg or "constructing hnsw segment reader" in msg:
            print(f"[복구] {collection_name} 인덱스 손상 감지. 컬렉션 재생성 후 재시도합니다.")
            _reset_collection(collection_name)
            store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                client=_get_client(),
            )
            store.add_documents(docs, ids=ids)
        else:
            raise


def _get_collection_count(collection_name: str) -> int:
    """컬렉션 문서 개수 반환. HNSW 에러 발생 시 exception 그대로 throw."""
    client = _get_client()
    return client.get_collection(collection_name).count()


def _collection_exists(collection_name: str) -> bool:
    """컬렉션 존재 여부만 확인 (문서 개수 세지 않음)."""
    client = _get_client()
    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        return False


def _reset_collection(collection_name: str) -> None:
    client = _get_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    # 일부 환경에서 삭제 후에도 로컬 세그먼트 파일이 남는 경우가 있어 정리
    base = Path(CHROMA_DIR)
    for p in base.glob(f"**/*{collection_name}*"):
        try:
            if p.is_file():
                p.unlink(missing_ok=True)
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass


DART_BM25_SIDECAR = PROJECT_ROOT / "dart_bm25.pkl"


def _load_dart_bm25_sidecar() -> list[Document] | None:
    """BM25 코퍼스를 사이드카 파일에서 로드.

    chromadb 1.5.x는 대용량 컬렉션의 전체 get()에서 backfill 레이스로 인한 HNSW 에러를
    비결정적으로 던진다(2026-06-19 디버깅). BM25는 모든 문서 텍스트가 필요하므로 이 불안정한
    경로 대신, ingest가 생성한 사이드카(dart_bm25.pkl)에서 안정적으로 로드한다.
    (벡터 검색 similarity_search는 안정적이라 chroma를 그대로 사용)
    """
    if not DART_BM25_SIDECAR.exists():
        return None
    try:
        with open(DART_BM25_SIDECAR, "rb") as f:
            rows = pickle.load(f)
        return [Document(page_content=r["text"], metadata=r.get("metadata") or {}) for r in rows]
    except Exception as e:
        print(f"[warn] BM25 사이드카 로드 실패, chroma fallback: {str(e)[:60]}")
        return None


def _load_all_docs(collection_name: str, retries: int = 8) -> list[Document]:
    # chromadb 1.5.x는 프로세스 cold-start의 첫 전체 get()에서 HNSW 인덱스를 backfill(컴팩션)하는데,
    # 이 과정이 아직 끝나지 않았으면 일시적으로 "Error loading hnsw index"를 던진다(비결정적).
    # 대응: ① count()로 backfill을 먼저 유도(워밍업) ② 실패 시 대기 후 재시도. 데이터는 삭제하지 않는다.
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            client = _get_client()
            collection = client.get_collection(collection_name)
            _ = collection.count()  # 워밍업: cold-start backfill 유도
            payload = collection.get(include=["documents", "metadatas"])
            docs = payload.get("documents") or []
            metas = payload.get("metadatas") or []
            return [Document(page_content=t, metadata=m or {}) for t, m in zip(docs, metas)]
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "hnsw" in msg or "backfill" in msg or "compactor" in msg or "segment reader" in msg:
                time.sleep(0.6 * (attempt + 1))
                continue
            raise
    raise last_err if last_err else RuntimeError("_load_all_docs 실패")


_SEGMENT_ERR_MARKERS = ("hnsw", "segment", "nothing found", "backfill", "compactor")


def _robust_query(
    collection_name: str,
    query_embedding: list[float],
    n_results: int,
    where: dict | None = None,
    retries: int = 5,
) -> list[Document]:
    """langchain Chroma 래퍼 대신 chromadb collection.query를 직접 호출(안정적).

    langchain Chroma의 similarity_search는 다중 컬렉션·반복 쿼리·장기 실행에서 세그먼트
    리더가 stale해져 'Nothing found on disk' 등을 던졌다(특히 작은 kam 컬렉션). 직접 쿼리는
    안정적이며, 그래도 세그먼트 에러가 나면 클라이언트를 새로 열어 재시도한다(2026-06-20 디버깅).
    """
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            col = _get_client().get_collection(collection_name)
            kwargs: dict = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
                "include": ["documents", "metadatas"],
            }
            if where:
                kwargs["where"] = where
            res = col.query(**kwargs)
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            return [Document(page_content=t, metadata=m or {}) for t, m in zip(docs, metas)]
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if any(marker in msg for marker in _SEGMENT_ERR_MARKERS):
                _reset_client()
                time.sleep(0.5 * (attempt + 1))
                continue
            raise
    raise last_err if last_err else RuntimeError("_robust_query 실패")


def _rrf(doc_lists: list[list[Document]], k: int = 60) -> list[Document]:
    """Reciprocal Rank Fusion: 여러 랭킹 리스트를 순위 기반 점수로 융합."""
    rrf_score: dict = {}
    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list):
            content = doc.page_content
            score = 1.0 / (rank + k)
            if content in rrf_score:
                rrf_score[content]["score"] += score
            else:
                rrf_score[content] = {"score": score, "doc": doc}
    return [item["doc"] for item in sorted(rrf_score.values(), key=lambda x: x["score"], reverse=True)]


from typing import Any
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# 회사/산업 감지는 industries.py의 전체 산업 레지스트리를 단일 진실 공급원으로 사용한다
# (industries.detect_industries_and_companies). 산업을 추가하려면 industries.py만 고치면 된다.


def _interleave(doc_lists: list[list[Document]], limit: int) -> list[Document]:
    """여러 리스트를 라운드로빈으로 섞어 각 리스트(=각 기업)의 대표성을 보장하며 합친다."""
    out: list[Document] = []
    seen: set[str] = set()
    for rank in range(max((len(dl) for dl in doc_lists), default=0)):
        for dl in doc_lists:
            if rank < len(dl):
                d = dl[rank]
                if d.page_content not in seen:
                    seen.add(d.page_content)
                    out.append(d)
                    if len(out) >= limit:
                        return out
    return out


def _enrich(docs: list[Document]) -> list[Document]:
    """BM25 키워드 매칭 강화를 위해 [기업 연도 섹션] 메타데이터를 본문 앞에 결합."""
    out = []
    for d in docs:
        company = d.metadata.get("company", "")
        year = d.metadata.get("year", "")
        section = d.metadata.get("section", "")
        prefix = f"[{company} {year}년 {section}] " if company else ""
        out.append(Document(page_content=prefix + (d.page_content or ""), metadata=d.metadata))
    return out


_YEAR_IN_QUERY_RE = re.compile(r"(20\d{2})\s*년")

# _company_docs가 매 요청마다 쓸 수 있도록 미리 만들어둘 회사(+연도)별 BM25 상한.
# _get_relevant_documents의 per(최대 50)를 커버할 만큼 넉넉히 잡고, 실제로는 그때그때 k로 슬라이싱한다.
_DART_BM25_PRECOMPUTE_K = 50


def _build_per_company_bm25(bm25_docs: list[Document]) -> tuple[dict[tuple[str, str], Any], dict[str, Any]]:
    """DART 전체 BM25 코퍼스를 (기업, 연도)별 / 기업별로 미리 나눠 BM25Retriever를 한 번만 구축한다.

    [응답 속도] 예전엔 질의가 들어올 때마다(비교 질문이면 기업 수만큼 반복해서) _company_docs
    안에서 BM25Retriever.from_documents()로 토큰화·인덱스 구축을 매번 새로 했다. 이게 답변
    생성이 1분 넘게 걸리는 원인 중 하나였다(2026-07-26 진단). 서버 시작 시(get_dart_retriever
    호출 시점) 딱 1번만 구축해두고, 요청마다는 조회만 하도록 바꾼다.
    """
    from collections import defaultdict
    from langchain_community.retrievers import BM25Retriever

    by_company_year: dict[tuple[str, str], list[Document]] = defaultdict(list)
    by_company: dict[str, list[Document]] = defaultdict(list)
    for d in bm25_docs:
        company = d.metadata.get("company")
        year = d.metadata.get("year")
        if not company:
            continue
        by_company[company].append(d)
        if year:
            by_company_year[(company, year)].append(d)

    company_year_bm25 = {
        key: BM25Retriever.from_documents(_enrich(docs), k=_DART_BM25_PRECOMPUTE_K)
        for key, docs in by_company_year.items()
    }
    company_bm25 = {
        company: BM25Retriever.from_documents(_enrich(docs), k=_DART_BM25_PRECOMPUTE_K)
        for company, docs in by_company.items()
    }
    return company_year_bm25, company_bm25


class DynamicDartRetriever(BaseRetriever):
    collection_name: str
    embeddings: Any
    global_bm25_docs: list
    global_bm25_retriever: Any
    company_year_bm25: dict
    company_bm25: dict

    def _latest_year_for(self, company: str) -> str | None:
        years = [d.metadata.get("year") for d in self.global_bm25_docs if d.metadata.get("company") == company]
        years = [y for y in years if y]
        return max(years) if years else None

    def _company_docs(self, query: str, qv: list[float], company: str, k: int, year: str | None) -> list[Document]:
        """단일 기업(+가능하면 단일 연도)으로 필터링한 벡터+BM25 하이브리드 결과(RRF).

        [연도 정합성] 예전엔 기업별로 독립적으로 최고점 청크만 뽑아서, "A와 B를 비교해줘" 질문에
        A는 2024년 자료가, B는 2025년 자료가 섞여 나오는 문제가 있었다(2026-07-26 진단 — 비교
        질문인데 회계연도가 안 맞으면 감사 자료로서 무의미하다). 질의에 특정 연도가 없으면 그
        기업의 "가장 최근 연도"로 고정해서, 비교 대상 기업들의 회계연도를 자동으로 맞춘다.
        해당 연도 데이터가 실제로 없으면(수집 누락 등) 연도 제한을 풀고 폴백한다.
        """
        # [주의] chromadb는 where 딕셔너리에 최상위 키(연산자)가 정확히 1개여야 한다
        # ({"company": x, "year": y}처럼 암묵적 AND는 지원 안 함 — "$and"로 명시해야 함).
        bm25 = self.company_year_bm25.get((company, year)) if year else None
        if bm25 is not None:
            where: dict = {"$and": [{"company": company}, {"year": year}]}
        else:
            where = {"company": company}
            bm25 = self.company_bm25.get(company)

        semantic = _robust_query(self.collection_name, qv, k, where=where)
        if bm25 is not None:
            return _rrf([semantic, bm25.invoke(query)[:k]])
        return semantic

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        _, positive, negated = industries.detect_industries_and_companies(query)
        qv = self.embeddings.embed_query(query)

        if positive:
            # [다중 기업] 각 기업별로 따로 검색해 라운드로빈으로 합쳐 *모든 기업의 대표성*을 보장.
            # (기존엔 첫 기업만 필터링해 비교 질문에서 나머지 기업이 누락됐다 — 2026-06-20 Q3 개선)
            year_match = _YEAR_IN_QUERY_RE.search(query)
            explicit_year = year_match.group(1) if year_match else None
            print(f"[DynamicDartRetriever] 대상 기업 {positive} 감지 (연도: {explicit_year or '기업별 최신'}). 기업별 균형 검색.")
            per = max(15, 50 // len(positive))
            per_company = [
                self._company_docs(query, qv, comp, per, explicit_year or self._latest_year_for(comp))
                for comp in positive
            ]
            return _interleave(per_company, limit=30)

        # 포함 대상이 없음 → 전역 검색. 단, 제외 대상(예: "삼천당이 아닌")이 있으면 필터로 뺀다.
        where = {"company": {"$nin": negated}} if negated else None
        if negated:
            print(f"[DynamicDartRetriever] 제외 기업 {negated} → 해당 기업 제외 전역 검색.")
        else:
            print("[DynamicDartRetriever] 기업명 미감지. 전역 검색 수행.")
        semantic_docs = _robust_query(self.collection_name, qv, 30, where=where)
        if self.global_bm25_retriever:
            bm25_docs = self.global_bm25_retriever.invoke(query)
            if negated:  # BM25 결과에서도 제외 기업 제거
                bm25_docs = [d for d in bm25_docs if d.metadata.get("company") not in negated]
            combined = _rrf([semantic_docs, bm25_docs])
        else:
            combined = semantic_docs
        return combined[:20]


class SimpleHybridRetriever(BaseRetriever):
    """kifrs용 하이브리드 리트리버. 직접 chromadb 쿼리(_robust_query) + BM25를 RRF로 융합.

    기존 EnsembleRetriever + langchain Chroma.as_retriever는 장기 실행에서 세그먼트 stale
    오류('Nothing found on disk')를 냈으므로, 안정적인 직접 쿼리 경로로 대체했다(2026-06-20).
    """
    collection_name: str
    embeddings: Any
    bm25_retriever: Any

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        qv = self.embeddings.embed_query(query)
        semantic_docs = _robust_query(self.collection_name, qv, 50)
        bm25_docs = self.bm25_retriever.invoke(query) if self.bm25_retriever else []
        return _rrf([semantic_docs, bm25_docs])[:30]


class DynamicGuidebookRetriever(BaseRetriever):
    """다산업 통합 guidebook(KAM+가이드북) 컬렉션용. DynamicDartRetriever와 동일한 패턴으로,
    질의에서 산업을 감지해 해당 산업 문서로 필터링한 하이브리드(BM25+Semantic RRF) 검색을 수행한다.
    산업 미감지 시 SimpleHybridRetriever와 동일하게 전역 검색으로 폴백한다.
    (K-IFRS 산업공통 기준서는 별도의 kifrs 컬렉션에 있으므로 여기서는 산업 필터만 신경쓰면 된다.)
    """
    collection_name: str
    embeddings: Any
    global_bm25_docs: list
    global_bm25_retriever: Any

    def _industry_docs(self, query: str, qv: list[float], industry: str, k: int) -> list[Document]:
        """단일 산업으로 필터링한 벡터+BM25 하이브리드 결과(RRF)."""
        semantic = _robust_query(self.collection_name, qv, k, where={"industry": industry})
        ind_bm25_docs = [d for d in self.global_bm25_docs if d.metadata.get("industry") == industry]
        if ind_bm25_docs:
            bm25 = BM25Retriever.from_documents(_enrich(ind_bm25_docs), k=k)
            return _rrf([semantic, bm25.invoke(query)])
        return semantic

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        detected, _, _ = industries.detect_industries_and_companies(query)
        qv = self.embeddings.embed_query(query)

        if detected:
            # [다중 산업] 비교 질문에서도 산업별 대표성을 보장하려 산업별로 따로 검색해 라운드로빈으로 합친다
            # (DynamicDartRetriever의 다중 기업 처리와 동일한 패턴).
            targets = sorted(detected)
            print(f"[DynamicGuidebookRetriever] 대상 산업 {targets} 감지. 산업별 균형 검색.")
            per = max(15, 40 // len(targets))
            per_industry = [self._industry_docs(query, qv, ind, per) for ind in targets]
            return _interleave(per_industry, limit=24)

        print("[DynamicGuidebookRetriever] 산업 미감지. 전역 검색 수행.")
        semantic_docs = _robust_query(self.collection_name, qv, 30)
        if self.global_bm25_retriever:
            bm25_docs = self.global_bm25_retriever.invoke(query)
            combined = _rrf([semantic_docs, bm25_docs])
        else:
            combined = semantic_docs
        return combined[:20]


def _make_base_hybrid_retriever(collection_name: str, embeddings: OpenAIEmbeddings, bm25_docs: list[Document]) -> BaseRetriever:
    if not bm25_docs:
        raise ValueError(f"{collection_name} 컬렉션에 검색 가능한 문서가 없습니다.")
    bm25 = BM25Retriever.from_documents(_enrich(bm25_docs), k=50)
    return SimpleHybridRetriever(collection_name=collection_name, embeddings=embeddings, bm25_retriever=bm25)


# FlashrankRerank의 관련도 점수(0~1) 하한선. 이 점수 미만인 청크는 컨텍스트에서 원천 제외되어
# "억지로 관련 있는 척 끼워맞추는" 할루시네이션의 소스를 줄인다. 하이브리드(RRF)는 등수만 반환해
# 컷오프를 걸 수 없지만, 2단계 Cross-Encoder 리랭커는 절대 점수를 반환하므로 여기서 적용한다.
# 0.02는 "명백히 무관한" 꼬리 결과만 걸러내는 보수적 초기값 — run_benchmark.py로 튜닝 권장.
FLASHRANK_SCORE_THRESHOLD = 0.02


def _build_2stage(
    base: BaseRetriever, rerank_top_n: int = 15, score_threshold: float = FLASHRANK_SCORE_THRESHOLD
) -> BaseRetriever:
    if ContextualCompressionRetriever is None or FlashrankRerank is None:
        print("[경고] FlashRank 미설치: 1단계 하이브리드 검색만 사용")
        return base
    compressor = FlashrankRerank(top_n=rerank_top_n, score_threshold=score_threshold)
    return ContextualCompressionRetriever(base_retriever=base, base_compressor=compressor)


def _ingest_docs(collection_name: str, embeddings: OpenAIEmbeddings, docs_to_ingest: list[Document]) -> None:
    """배치 임베딩 + chroma 저장만 수행(리트리버는 만들지 않음). DynamicGuidebookRetriever처럼
    SimpleHybridRetriever가 아닌 별도 리트리버로 서빙할 컬렉션의 lazy ingest에 사용."""
    for i in range(0, len(docs_to_ingest), EMBED_BATCH_SIZE):
        save_to_chroma(collection_name, embeddings, docs_to_ingest[i : i + EMBED_BATCH_SIZE])
        if i + EMBED_BATCH_SIZE < len(docs_to_ingest):
            time.sleep(EMBED_BATCH_DELAY)


def _make_retriever(collection_name: str, embeddings: OpenAIEmbeddings, docs_to_ingest: list[Document] | None = None) -> BaseRetriever:
    if docs_to_ingest:
        _ingest_docs(collection_name, embeddings, docs_to_ingest)
        bm25_docs = docs_to_ingest
    else:
        # _load_all_docs는 내부적으로 워밍업+재시도를 수행한다. 실패해도 컬렉션을 삭제(reset)하지
        # 않는다 — 일시적 backfill 오류에 데이터를 영구 삭제하던 과거 동작이 치명적이었기 때문.
        bm25_docs = _load_all_docs(collection_name)

    # 컬렉션이 비어 있으면 retriever 생성 불가이므로 ingest 필요를 명확히 반환하기 위해
    # 빈 BM25 소스일 때는 최소 placeholder를 두지 않고 상위 함수에서 재ingest하도록 유도
    base = _make_base_hybrid_retriever(collection_name, embeddings, bm25_docs if bm25_docs else [])
    return _build_2stage(base, rerank_top_n=10)


def get_kifrs_retriever(embeddings: OpenAIEmbeddings) -> BaseRetriever:
    collection = "kifrs"

    # 컬렉션 없으면 PDF에서 초기화 (산업 공통 기준서이므로 industry="공통")
    if not _collection_exists(collection):
        print("[info] KIFRS 컬렉션 초기화 중...")
        raw: list[Document] = []
        for filename, source_id in industries.COMMON_KIFRS_PDFS:
            p = PROJECT_ROOT / "sources" / filename
            if not p.exists():
                p = PROJECT_ROOT / filename  # 과거 경로 호환(최초 3개 기준서는 루트에도 있음)
            if p.exists():
                raw.extend(_load_pdf(p, source_id, industry="공통", doc_type="standard"))
            else:
                print(f"[warn] {filename} 파일을 찾을 수 없습니다")
        _make_retriever(collection, embeddings, docs_to_ingest=_chunk(raw))
        print(f"[OK] KIFRS 초기화 완료: {len(raw)}개 문서")

    # 리트리버 로드 (_load_all_docs가 워밍업+재시도로 cold-start HNSW 오류를 처리. reset 없음)
    return _make_retriever(collection, embeddings)


def get_guidebook_retriever(embeddings: OpenAIEmbeddings) -> BaseRetriever:
    """산업별 KAM/가이드북 통합 컬렉션(舊 'kam'). industries.py 레지스트리의 5개 산업
    가이드북 PDF를 모두 인입하고, 산업별 필터링이 가능한 DynamicGuidebookRetriever로 서빙한다."""
    collection = "guidebook"

    # 컬렉션 없으면 전 산업 가이드북 PDF에서 초기화
    freshly_ingested: list[Document] | None = None
    if not _collection_exists(collection):
        print("[info] guidebook 컬렉션 초기화 중 (5개 산업)...")
        raw: list[Document] = []
        for industry in industries.INDUSTRIES.values():
            for filename, source_id, doc_type in industry.guidebook_pdfs:
                p = PROJECT_ROOT / "sources" / filename
                if not p.exists():
                    p = PROJECT_ROOT / filename  # 과거 경로 호환
                if p.exists():
                    raw.extend(_load_pdf(p, source_id, industry=industry.key, doc_type=doc_type))
                else:
                    print(f"[warn] {filename} 파일을 찾을 수 없습니다 (industry={industry.key})")
        freshly_ingested = _chunk(raw)
        _ingest_docs(collection, embeddings, freshly_ingested)
        print(f"[OK] guidebook 초기화 완료: {len(raw)}개 문서")

    # DynamicGuidebookRetriever로 서빙 (산업 감지 시 where={"industry":...} 필터링).
    # [중요] 방금 막 ingest한 직후라면 chroma를 즉시 재조회하지 않고 메모리에 있는 chunk를
    # 그대로 BM25 코퍼스로 쓴다 — _make_retriever와 동일한 패턴. chromadb 1.5.x는 대량 쓰기
    # 직후의 즉시 재조회에서 backfill이 아직 안 끝나 빈 결과를 "에러 없이" 반환할 수 있어
    # (코드 상단 HNSW 관련 디버깅 노트 참고), 재시도로도 못 잡는다.
    bm25_docs = freshly_ingested if freshly_ingested is not None else _load_all_docs(collection)
    if not bm25_docs:
        raise ValueError("guidebook 컬렉션에 검색 가능한 문서가 없습니다.")

    from langchain_community.retrievers import BM25Retriever
    global_bm25 = BM25Retriever.from_documents(_enrich(bm25_docs), k=30)
    base = DynamicGuidebookRetriever(
        collection_name=collection,
        embeddings=embeddings,
        global_bm25_docs=bm25_docs,
        global_bm25_retriever=global_bm25,
    )
    return _build_2stage(base, rerank_top_n=10)


def get_dart_retriever(embeddings: OpenAIEmbeddings) -> BaseRetriever | None:
    # 컬렉션 이름: 오염된 'dart' 대신 'dart_docs' 사용 (2026-06-19 HNSW 영속화 디버깅 참고)
    collection = "dart_docs"
    
    # Step 1: 컬렉션 존재 여부 확인
    if not _collection_exists(collection):
        print(f"[info] DART 컬렉션이 없습니다. 초기화 필요: python dart_ingest.py 실행")
        return None
    
    # Step 2: 컬렉션 로드 시도 (HNSW 에러는 _load_all_docs가 재시도로 처리)
    # [중요] 과거엔 HNSW 에러 시 _reset_collection으로 컬렉션을 삭제했는데, 이는 일시적
    # backfill 오류에도 데이터를 영구 삭제하는 치명적 동작이었다. 이제 삭제하지 않고
    # None만 반환해 데이터를 보존한다(2026-06-19 디버깅).
    # BM25 코퍼스: 안정적인 사이드카 파일 우선, 없으면 chroma full-get으로 fallback
    bm25_docs = _load_dart_bm25_sidecar()
    if bm25_docs is not None:
        print(f"[OK] DART BM25 사이드카 로드: {len(bm25_docs)}개 문서")
    else:
        try:
            bm25_docs = _load_all_docs(collection)
            print(f"[OK] DART 리트리버 로드 성공(chroma): {len(bm25_docs)}개 문서")
        except Exception as e:
            msg = str(e).lower()
            if "hnsw" in msg or "constructing hnsw segment reader" in msg or "backfill" in msg or "compactor" in msg:
                print(f"[!] DART HNSW 로드 실패(재시도 후에도): {str(e)[:80]}")
                print(f"[info] 데이터는 보존됨. dart_bm25.pkl 사이드카를 생성하면 안정화됩니다.")
                return None
            else:
                print(f"[ERROR] DART 로드 중 예기치 않은 에러: {e}")
                raise

    if not bm25_docs:
        print("[info] DART 컬렉션이 비어있습니다.")
        return None

    # Step 3: 리트리버 생성 (직접 chromadb 쿼리 기반 — langchain Chroma store 불필요)
    from langchain_community.retrievers import BM25Retriever
    global_bm25 = BM25Retriever.from_documents(_enrich(bm25_docs), k=30)
    # 기업(+연도)별 BM25도 여기서(서버 시작 시 1회) 미리 구축 — 요청마다 다시 만들지 않는다.
    company_year_bm25, company_bm25 = _build_per_company_bm25(bm25_docs)
    base = DynamicDartRetriever(
        collection_name=collection,
        embeddings=embeddings,
        global_bm25_docs=bm25_docs,
        company_year_bm25=company_year_bm25,
        company_bm25=company_bm25,
        global_bm25_retriever=global_bm25,
    )
    print("[OK] DART 하이브리드 리트리버 생성 완료")
    # rerank_top_n=10은 원래 기업 1곳 기준 질문에 맞춰 튜닝된 값이라, 2개 이상 기업을
    # 비교할 때 기업당 몇 개 청크밖에 안 남아 리스부채·사용권자산처럼 여러 하위 표로
    # 나뉜 주석의 세부 항목이 잘려나갔다(2026-07-26 진단). 다산업 확장 이후 기본값을 20으로 상향.
    return _build_2stage(base, rerank_top_n=20)
