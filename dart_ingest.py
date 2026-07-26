"""DART ingest with broader note extraction and higher document volume."""

from __future__ import annotations

import argparse
import html
import io
import os
import pickle
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from industries import INDUSTRIES, Industry, TargetCompany

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

DART_API_BASE = "https://opendart.fss.or.kr/api"
DART_VIEWER_BASE = "https://dart.fss.or.kr"
# [근본 원인] 한글 경로에서 chromadb Rust HNSW 로더가 실패하므로 벡터 DB는 ASCII 절대경로에 둔다.
CHROMA_DIR = os.path.join(os.path.expanduser("~"), "rag_chroma_data", "chroma_db_v2")
if not CHROMA_DIR.isascii():
    CHROMA_DIR = str(BASE_DIR / "chroma_db_v2")
# 컬렉션 이름: 'dart'를 삭제→재생성 반복하며 누적된 잔여 상태가 HNSW 인덱스 영속화를
# 깨뜨려(2026-06-19 디버깅), 오염되지 않은 'dart_docs' 이름으로 이전함.
COLLECTION = "dart_docs"


NOTE_PARENT_KEYWORDS = ["일반사항", "회사 개요", "중요한 회계정책", "주석", "사업의 내용", "연구개발활동"]
NOTEISH_TITLE_RE = re.compile(
    r"(주석|회계정책|유의사항|중요한\s*회계|재무제표|부문정보|추가정보|보충정보|사업의\s*내용|연구개발)"
)

# 하위 노드를 통째로 포함하는 "상위 폴더(컨테이너)" 노드는 본문이 과대(수 MB)하므로 제외.
# 예: "II. 사업의 내용"(295K), "III. 재무에 관한 사항"(7M), "3. 연결재무제표 주석"(2.8M)
CONTAINER_TITLE_RE = re.compile(r"^\s*[IVXLCDM]+\.\s")  # 로마숫자 대제목(I. II. III. ...)
MAX_CONTAINER_LENGTH = 1_500_000  # 이보다 큰 노드는 서브트리 묶음으로 간주하여 제외
ANNUAL_REPORT_YEAR_RE = re.compile(r"사업보고서\s*\((\d{4})\.\d{2}\)")

CORP_CODE_MAP: dict[str, str] = {}
CORP_NAME_MAP: dict[str, tuple[str, str, str]] = {}

TARGET_REPORT_YEARS = ("2025", "2024")
REPORT_LOOKBACK = len(TARGET_REPORT_YEARS)
MAX_SECTION_NODES = 30
SECTION_CHUNK_SIZE = 1000
SECTION_CHUNK_OVERLAP = 150
TOPIC_WINDOW_LEFT = 800
TOPIC_WINDOW_RIGHT = 1200
MERGED_BUNDLE_CHUNK_SIZE = 1200
MERGED_BUNDLE_OVERLAP = 150


@dataclass
class DocNode:
    dcm_no: str
    ele_id: str
    offset: str
    length: str
    dtd: str
    title: str

    @property
    def length_int(self) -> int:
        try:
            return int(self.length)
        except Exception:
            return 0


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": "https://dart.fss.or.kr/",
        }
    )
    return session


SESSION: requests.Session | None = None


def _get_session() -> requests.Session:
    global SESSION
    if SESSION is None:
        SESSION = _build_session()
    return SESSION


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").strip()


def _normalize_name(text: str) -> str:
    return re.sub(r"\s+", "", _normalize(text))


def _load_corp_reference_map(api_key: str) -> tuple[dict[str, str], dict[str, tuple[str, str, str]]]:
    print("[info] 기업코드 정보 로드 시작 (최대 30초)")
    try:
        r = _get_session().get(f"{DART_API_BASE}/corpCode.xml", params={"crtfc_key": api_key}, timeout=30)
        r.raise_for_status()
    except requests.exceptions.Timeout:
        print("[error] 기업코드 API timeout - 네트워크 확인 후 재시도하세요")
        raise
    except Exception as e:
        print(f"[error] 기업코드 API 호출 실패: {e}")
        raise
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        xml_bytes = zf.read("CORPCODE.xml")

    root = ET.fromstring(xml_bytes)
    by_stock: dict[str, str] = {}
    by_name: dict[str, tuple[str, str, str]] = {}

    for item in root.findall("list"):
        corp_name = _normalize(item.findtext("corp_name") or "")
        stock_code = (item.findtext("stock_code") or "").strip()
        corp_code = (item.findtext("corp_code") or "").strip()
        if corp_code and stock_code:
            by_stock[stock_code] = corp_code
            by_name[_normalize_name(corp_name)] = (corp_code, stock_code, corp_name)

    return by_stock, by_name


def _ensure_corp_reference_map(api_key: str) -> bool:
    global CORP_CODE_MAP, CORP_NAME_MAP
    if CORP_CODE_MAP and CORP_NAME_MAP:
        return True
    try:
        CORP_CODE_MAP, CORP_NAME_MAP = _load_corp_reference_map(api_key)
        return True
    except Exception as e:
        print(f"[fail] corpCode load failed: {e}")
        return False


def _find_company_reference(
    api_key: str,
    company_name: str,
    stock_code: str | None = None,
    aliases: tuple[str, ...] = (),
) -> tuple[str, str, str] | None:
    if not _ensure_corp_reference_map(api_key):
        return None

    if stock_code:
        corp_code = CORP_CODE_MAP.get(stock_code)
        if corp_code:
            return corp_code, stock_code, company_name

    candidates = [company_name, *aliases]
    for candidate in candidates:
        ref = CORP_NAME_MAP.get(_normalize_name(candidate))
        if ref:
            return ref

    normalized_candidates = {_normalize_name(name) for name in candidates}
    for corp_name_key, ref in CORP_NAME_MAP.items():
        if any(candidate in corp_name_key or corp_name_key in candidate for candidate in normalized_candidates):
            return ref

    return None


def _get_recent_reports(api_key: str, corp_code: str, limit: int = REPORT_LOOKBACK) -> list[dict[str, str]]:
    target_receipt_years = {str(int(year) + 1) for year in TARGET_REPORT_YEARS}
    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "pblntf_ty": "A",
        "bgn_de": f"{min(target_receipt_years)}0101",
        "end_de": f"{max(target_receipt_years)}1231",
        "page_count": "50",
    }
    try:
        r = _get_session().get(f"{DART_API_BASE}/list.json", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "000":
            print(f"[fail] list.json status={data.get('status')} message={data.get('message')}")
            return []

        rows = data.get("list", []) or []
        annual: list[dict[str, str]] = []
        for row in rows:
            report_nm = row.get("report_nm") or ""
            if "사업보고서" not in report_nm:
                continue
            if "반기" in report_nm or "분기" in report_nm:
                continue
            match = ANNUAL_REPORT_YEAR_RE.search(report_nm)
            if not match:
                continue
            report_year = match.group(1)
            if report_year not in TARGET_REPORT_YEARS:
                continue
            if not row.get("rcept_no"):
                continue
            annual.append(
                {
                    "rcp_no": row.get("rcept_no", ""),
                    "report_nm": report_nm,
                    "rcept_dt": row.get("rcept_dt", ""),
                    "report_year": report_year,
                    "is_correction": "정정" in report_nm,
                }
            )

        # [버그 수정] 타깃 연도별로 1개씩 선택해 연도 커버리지를 보장한다.
        # 기존엔 단순 [:limit]라서, 한 연도의 (원본+기재정정) 2건이 슬롯을 모두 차지해
        # 다른 연도(예: 2024)가 누락됐다. 또한 기재정정본은 일부 섹션만 담는 경우가 많아
        # 동일 연도에선 "원본(non-correction)"을 우선하고, 같은 종류면 최신 접수분을 택한다.
        best_by_year: dict[str, dict[str, str]] = {}
        for item in annual:
            year = item["report_year"]
            current = best_by_year.get(year)
            if current is None:
                best_by_year[year] = item
                continue
            # 원본 > 정정 우선
            if current["is_correction"] and not item["is_correction"]:
                best_by_year[year] = item
            elif current["is_correction"] == item["is_correction"] and item["rcept_dt"] > current["rcept_dt"]:
                best_by_year[year] = item

        selected = [best_by_year[year] for year in TARGET_REPORT_YEARS if year in best_by_year]
        return selected[:limit]
    except Exception as e:
        print(f"[fail] report list fetch failed: {e}")
        return []


def _fetch_left_panel(rcp_no: str) -> str:
    time.sleep(1)  # Rate limiting 대비
    r = _get_session().get(
        f"{DART_VIEWER_BASE}/dsaf001/main.do",
        params={"rcpNo": rcp_no, "leftFrameDiv": "X"},
        timeout=20,
    )
    r.raise_for_status()
    return r.text


def _decode_js_value(value: str) -> str:
    value = value.replace('\\"', '"').replace("\\n", " ").replace("\\t", " ")
    value = html.unescape(value)
    return _normalize(value)


def _parse_doc_nodes(left_html: str) -> list[DocNode]:
    # DART 좌측 패널 JS는 `var node1 = {}; node1['text']=...; ...` 형태로 노드를 선언하는데,
    # node1/node2/node3 변수명을 "트리 깊이"별로 재사용한다(같은 변수명을 형제 노드마다 덮어씀).
    # 따라서 변수명(idx)으로 버킷팅하면 마지막 값만 남아 노드가 소실된다.
    # 올바른 방법: `var nodeN = {}` 선언을 경계로 순차 분할하여 각 세그먼트를 1개 노드로 본다.
    field_pattern = re.compile(r"node\d+\['(?P<key>[^']+)'\]\s*=\s*\"(?P<value>.*?)\";", re.DOTALL)
    segments = re.split(r"var\s+node\d+\s*=\s*\{\}\s*;", left_html)

    nodes: list[DocNode] = []
    for segment in segments:
        bucket: dict[str, str] = {}
        for match in field_pattern.finditer(segment):
            bucket[match.group("key")] = _decode_js_value(match.group("value"))

        required = ["dcmNo", "eleId", "offset", "length", "dtd", "text"]
        if not all(key in bucket for key in required):
            continue
        nodes.append(
            DocNode(
                dcm_no=bucket["dcmNo"],
                ele_id=bucket["eleId"],
                offset=bucket["offset"],
                length=bucket["length"],
                dtd=bucket["dtd"],
                title=bucket["text"],
            )
        )
    return nodes


def _find_notes_parent_node(nodes: list[DocNode]) -> DocNode | None:
    if not nodes:
        return None

    max_len = max(node.length_int for node in nodes) or 1
    keyword_re = re.compile("|".join(re.escape(keyword) for keyword in NOTE_PARENT_KEYWORDS))
    notes_anchor_re = re.compile(r"(재무제표\s*)?주석")

    anchor_idx: int | None = None
    for index, node in enumerate(nodes):
        if notes_anchor_re.search(_normalize(node.title)):
            anchor_idx = index
            break

    scored: list[tuple[float, DocNode]] = []
    for index, node in enumerate(nodes):
        title = _normalize(node.title)
        ratio = node.length_int / max_len
        score = ratio
        if keyword_re.search(title):
            score += 0.5
        if notes_anchor_re.search(title):
            score += 0.35
        if anchor_idx is not None and index >= anchor_idx:
            score += 0.15
        if ratio >= 0.08 or notes_anchor_re.search(title) or keyword_re.search(title):
            scored.append((score, node))

    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]
    return max(nodes, key=lambda item: item.length_int)


def _is_container_node(node: DocNode) -> bool:
    """하위 노드를 통째로 묶은 상위 폴더(컨테이너) 노드인지 판별. 컨테이너는 본문이 과대해 제외한다."""
    title = _normalize(node.title)
    if CONTAINER_TITLE_RE.search(title):
        return True
    if node.length_int > MAX_CONTAINER_LENGTH:
        return True
    return False


def _select_note_nodes(nodes: list[DocNode], section_whitelist: tuple[str, ...]) -> list[DocNode]:
    """[옵션 A+B] 화이트리스트 기반 섹션 선택.

    기존 길이/빈도 휴리스틱은 분량 큰 곁다리 섹션(특수관계자·위험관리 등)을 선호해
    정작 핵심 섹션을 누락시켰다(2026-06-19 진단). 이제 산업별 section_whitelist(industries.py,
    산업마다 다른 회계 이슈에 맞춰 다름) 키워드에 걸리는 리프 노드만 선택한다.
    """
    if not nodes:
        return []

    selected: list[DocNode] = []
    seen: set[str] = set()
    for node in nodes:
        title = _normalize(node.title)
        if not title or title in seen:
            continue
        if _is_container_node(node):
            continue
        if any(keyword in title for keyword in section_whitelist):
            seen.add(title)
            selected.append(node)
            if len(selected) >= MAX_SECTION_NODES:
                break
    return selected


def _html_to_markdown(soup: BeautifulSoup) -> str:
    # Convert tables to markdown
    for table in soup.find_all("table"):
        markdown_table = []
        rows = table.find_all("tr")
        is_first_row = True
        for row in rows:
            cols = row.find_all(["th", "td"])
            if not cols:
                continue
            # Extract text, replace newlines and pipes to avoid breaking markdown table
            row_text = [col.get_text(separator=" ", strip=True).replace("\n", " ").replace("|", ",") for col in cols]
            markdown_table.append("| " + " | ".join(row_text) + " |")
            if is_first_row:
                markdown_table.append("|" + "|".join(["---"] * len(row_text)) + "|")
                is_first_row = False
        
        if markdown_table:
            # Insert markdown string before the table
            table.insert_before(soup.new_string("\n\n" + "\n".join(markdown_table) + "\n\n"))
        table.decompose()
        
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def _fetch_section_text(rcp_no: str, node: DocNode) -> str:
    time.sleep(1)  # Rate limiting 대비
    params = {
        "rcpNo": rcp_no,
        "dcmNo": node.dcm_no,
        "eleId": node.ele_id,
        "offset": node.offset,
        "length": node.length,
        "dtd": node.dtd,
    }
    r = _get_session().get(f"{DART_VIEWER_BASE}/report/viewer.do", params=params, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return _html_to_markdown(soup)


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = _normalize(text)
    if not text:
        return []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def _merge_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    sorted_spans = sorted(spans, key=lambda item: item[0])
    if not sorted_spans:
        return []

    merged = [sorted_spans[0]]
    for start, end in sorted_spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _extract_topic_chunks(full_text: str, topic: str, topic_keywords: dict[str, tuple[str, ...]]) -> list[str]:
    keywords = topic_keywords.get(topic, (topic,))
    spans: list[tuple[int, int]] = []
    for keyword in keywords:
        for match in re.finditer(re.escape(keyword), full_text):
            start = max(0, match.start() - TOPIC_WINDOW_LEFT)
            end = min(len(full_text), match.end() + TOPIC_WINDOW_RIGHT)
            spans.append((start, end))
    if not spans:
        return []

    chunks: list[str] = []
    for start, end in _merge_spans(spans):
        text = full_text[start:end].strip()
        if text:
            chunks.append(text)
    return chunks


def _build_doc_id(meta: dict[str, str], idx: int) -> str:
    company = re.sub(r"\s+", "", meta.get("company", "unknown"))
    year = meta.get("year", "unknown")
    section = re.sub(r"\s+", "_", meta.get("section", "unknown"))
    rcp_no = meta.get("rcpNo", "unknown")
    section_type = meta.get("section_type", "chunk")
    return f"{company}_{year}_{section}_{section_type}_{rcp_no}_{idx}"


def _save_to_chroma(embeddings: OpenAIEmbeddings, docs: list[Document]) -> None:
    if not docs:
        return

    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    ids = [_build_doc_id(doc.metadata, index) for index, doc in enumerate(docs)]
    try:
        store.delete(ids=ids)
    except Exception:
        pass
    store.add_documents(docs, ids=ids)


def _build_documents_for_report(
    industry: Industry,
    company_name: str,
    stock_code: str,
    corp_code: str,
    report: dict[str, str],
    sections: list[tuple[DocNode, str]],
    parent: DocNode | None,
) -> list[Document]:
    rcp_no = report["rcp_no"]
    report_nm = report.get("report_nm", "")
    report_dt = report.get("rcept_dt", "")
    year = report.get("report_year") or (report_dt[:4] if report_dt else "2025")
    docs: list[Document] = []

    merged_notes = "\n\n".join(text for _, text in sections)

    for node, text in sections:
        for idx, chunk in enumerate(_chunk_text(text, SECTION_CHUNK_SIZE, SECTION_CHUNK_OVERLAP)):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": f"DART_{company_name}_{year}",
                        "company": company_name,
                        "industry": industry.key,
                        "ticker": stock_code,
                        "corp_code": corp_code,
                        "rcpNo": rcp_no,
                        "report_nm": report_nm,
                        "report_dt": report_dt,
                        "year": year,
                        "section": node.title,
                        "chunk_index": str(idx),
                        "section_type": "note_section",
                        "source_url": f"{DART_VIEWER_BASE}/dsaf001/main.do?rcpNo={rcp_no}",
                        "node_title": parent.title if parent else "",
                    },
                )
            )

    # [옵션 C] notes_bundle(merged_notes 통합본) 저장은 제거.
    # 동일 내용을 note_section과 중복 저장해 검색 다양성을 떨어뜨리고 임베딩 비용을 2배로 늘렸다.
    # 단, 토픽 윈도우 추출을 위해 merged_notes 텍스트 자체는 아래에서 계속 사용한다.
    for topic in industry.accounting_topics:
        for idx, chunk in enumerate(_extract_topic_chunks(merged_notes, topic, industry.topic_keywords)):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": f"DART_{company_name}_{year}",
                        "company": company_name,
                        "industry": industry.key,
                        "ticker": stock_code,
                        "corp_code": corp_code,
                        "rcpNo": rcp_no,
                        "report_nm": report_nm,
                        "report_dt": report_dt,
                        "year": year,
                        "section": topic,
                        "chunk_index": str(idx),
                        "section_type": "topic_window",
                        "source_url": f"{DART_VIEWER_BASE}/dsaf001/main.do?rcpNo={rcp_no}",
                        "node_title": parent.title if parent else "",
                    },
                )
            )

    return docs


def _collect_sections_for_report(
    rcp_no: str, section_whitelist: tuple[str, ...]
) -> tuple[list[tuple[DocNode, str]], DocNode | None]:
    try:
        left_html = _fetch_left_panel(rcp_no)
    except Exception as e:
        print(f"[fail] left panel fetch failed: {e}")
        return [], None

    nodes = _parse_doc_nodes(left_html)
    if not nodes:
        print("[fail] node parsing failed: 0 nodes")
        return [], None

    parent = _find_notes_parent_node(nodes)
    selected_nodes = _select_note_nodes(nodes, section_whitelist)
    if not selected_nodes:
        print("[fail] no candidate note nodes found")
        return [], parent

    sections: list[tuple[DocNode, str]] = []
    for node in selected_nodes:
        try:
            text = _fetch_section_text(rcp_no, node)
            if text.strip():
                sections.append((node, text))
        except Exception as e:
            print(f"[warn] section fetch failed ({node.title}): {e}")
    return sections, parent


def ingest_company(
    api_key: str, embeddings: OpenAIEmbeddings, industry: Industry, target: TargetCompany
) -> list[Document]:
    """DART에서 target 기업의 최근 사업보고서를 수집해 chroma에 저장한다.
    반환값: 저장된 Document 리스트(BM25 사이드카 누적용). 실패/스킵 시 빈 리스트."""
    ref = _find_company_reference(api_key, target.company_name, target.stock_code, target.aliases)
    if not ref:
        print(f"[skip] {target.company_name}: corp_code lookup failed")
        return []

    corp_code, stock_code, resolved_name = ref
    reports = _get_recent_reports(api_key, corp_code)
    if not reports:
        print(f"[skip] {target.company_name}: no target annual report found")
        return []

    docs: list[Document] = []
    total_sections = 0
    total_topic_windows = 0

    for report_idx, report in enumerate(reports):
        if report_idx > 0:
            time.sleep(2)  # 보고서 간 대기 대비
        sections, parent = _collect_sections_for_report(report["rcp_no"], industry.section_whitelist)
        if not sections:
            print(f"[skip] {target.company_name}: no sections for rcpNo={report['rcp_no']}")
            continue
        docs.extend(
            _build_documents_for_report(industry, target.company_name, stock_code, corp_code, report, sections, parent)
        )
        total_sections += len(sections)
        total_topic_windows += sum(
            1
            for doc in docs
            if doc.metadata.get("section_type") == "topic_window" and doc.metadata.get("rcpNo") == report["rcp_no"]
        )

    if not docs:
        print(f"[skip] {target.company_name}: no documents extracted")
        return []

    _save_to_chroma(embeddings, docs)
    print(
        f"[ok] {target.company_name}: saved {len(docs)} docs "
        f"(industry={industry.key}, resolved={resolved_name}, reports={len(reports)}, "
        f"sections={total_sections}, topic_windows={total_topic_windows})"
    )
    return docs


DART_BM25_SIDECAR = Path(__file__).resolve().parent / "dart_bm25.pkl"


def _load_existing_sidecar() -> list[dict]:
    """기존 사이드카가 있으면 로드(다른 산업을 --industry로 나눠 돌릴 때 누적하기 위함)."""
    if not DART_BM25_SIDECAR.exists():
        return []
    try:
        with open(DART_BM25_SIDECAR, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[warn] 기존 dart_bm25.pkl 로드 실패, 새로 씁니다: {str(e)[:60]}")
        return []


def _write_bm25_sidecar(new_docs: list[Document]) -> None:
    """rag_engine._load_dart_bm25_sidecar()가 읽는 dart_bm25.pkl을 갱신한다.

    [버그 수정] 예전엔 이 사이드카를 쓰는 코드 자체가 없어서, rag_engine이 기대하던
    "안정적인 BM25 코퍼스"가 실제로는 만들어진 적이 없었다(주석만 있고 구현 누락).
    회사명+연도+섹션 기준으로 같은 문서를 덮어써서, --industry로 나눠 돌려도 중복 없이 누적된다.
    """
    rows_by_key: dict[str, dict] = {
        f"{r['metadata'].get('company')}|{r['metadata'].get('year')}|{r['metadata'].get('section')}|{r['metadata'].get('chunk_index')}|{r['metadata'].get('section_type')}": r
        for r in _load_existing_sidecar()
    }
    for doc in new_docs:
        m = doc.metadata
        key = f"{m.get('company')}|{m.get('year')}|{m.get('section')}|{m.get('chunk_index')}|{m.get('section_type')}"
        rows_by_key[key] = {"text": doc.page_content, "metadata": m}

    rows = list(rows_by_key.values())
    with open(DART_BM25_SIDECAR, "wb") as f:
        pickle.dump(rows, f)
    print(f"[OK] dart_bm25.pkl 사이드카 갱신: 총 {len(rows)}개 문서")


def main() -> None:
    parser = argparse.ArgumentParser(description="DART 다산업 공시 데이터 수집")
    parser.add_argument(
        "--industry",
        choices=sorted(INDUSTRIES.keys()),
        default=None,
        help="특정 산업만 수집(예: construction). 생략 시 전 산업(76개사 내외, 수 시간 소요 가능) 수집.",
    )
    args = parser.parse_args()

    target_industries: list[Industry] = (
        [INDUSTRIES[args.industry]] if args.industry else list(INDUSTRIES.values())
    )

    print(f"[시작] DART 데이터 수집 시작 (대상 산업: {[i.key for i in target_industries]})")
    dart_key = os.getenv("DART_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if not dart_key or dart_key == "your_dart_api_key_here":
        print("[error] DART_API_KEY is missing.")
        sys.exit(1)
    if not openai_key or openai_key == "your_openai_api_key_here":
        print("[error] OPENAI_API_KEY is missing.")
        sys.exit(1)

    print("[info] 임베딩 모델 초기화 중...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("[info] 임베딩 모델 준비 완료")

    all_targets = [(industry, company) for industry in target_industries for company in industry.companies]
    success = 0
    all_new_docs: list[Document] = []

    for idx, (industry, target) in enumerate(all_targets, 1):
        print(
            f"\n[진행] {idx}/{len(all_targets)}: [{industry.label}] {target.company_name} 처리 중...",
            flush=True,
        )
        try:
            docs = ingest_company(dart_key, embeddings, industry, target)
            if docs:
                success += 1
                all_new_docs.extend(docs)
            time.sleep(3)  # 회사 간 대기 대비 (rate limiting 방지)
        except Exception as e:
            print(f"[error] {target.company_name}: {e}")
            time.sleep(5)  # 에러 발생 시 더 대기

    if all_new_docs:
        _write_bm25_sidecar(all_new_docs)

    print(f"\n[완료] ingest complete: {success}/{len(all_targets)} companies")


if __name__ == "__main__":
    main()
