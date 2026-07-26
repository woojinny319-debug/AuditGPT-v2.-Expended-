"""Industry registry: single source of truth for multi-industry ingest & retrieval.

rag_engine.py와 dart_ingest.py가 모두 여기서 산업별 가이드북 PDF, DART 대상 기업,
DART 수집 토픽/키워드/섹션 화이트리스트를 가져온다. 산업을 추가/변경할 땐 이 파일만
고치면 된다.

[주의] prompts.py는 사용자 지시로 수정 대상에서 제외되었다. 따라서 SYSTEM_PROMPT는
여전히 제약·바이오 문구/32개사 리스트로 고정되어 있으며, 이 레지스트리의 산업 목록을
프롬프트에 동적으로 반영하지 않는다(검색/필터링 레이어에서만 다산업이 적용됨).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetCompany:
    company_name: str
    stock_code: str | None = None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Industry:
    key: str
    label: str
    # (filename in sources/, source_id for citations, doc_type: "kam" | "guidebook")
    guidebook_pdfs: tuple[tuple[str, str, str], ...]
    companies: tuple[TargetCompany, ...]
    accounting_topics: tuple[str, ...]
    topic_keywords: dict[str, tuple[str, ...]]
    section_whitelist: tuple[str, ...]


# 산업 공통 K-IFRS 기준서. industry="공통"으로 태깅되며 모든 산업 질의에 항상 노출된다.
COMMON_KIFRS_PDFS: tuple[tuple[str, str], ...] = (
    ("K-IFRS_제1002호_재고자산.pdf", "K-IFRS_1002"),
    ("K-IFRS_제1016호_유형자산.pdf", "K-IFRS_1016"),
    ("K-IFRS_제1036호_자산손상.pdf", "K-IFRS_1036"),
    ("K-IFRS_제1037호_충당부채_우발부채_우발자산.pdf", "K-IFRS_1037"),
    ("K-IFRS_제1038호_무형자산.pdf", "K-IFRS_1038"),
    ("K-IFRS_제1041호_농림어업.pdf", "K-IFRS_1041"),
    ("K-IFRS_제1109호_금융상품.pdf", "K-IFRS_1109"),
    ("K-IFRS_제1110호_연결재무제표.pdf", "K-IFRS_1110"),
    ("K-IFRS_제1115호_수익.pdf", "K-IFRS_1115"),
    ("K-IFRS_제1116호_리스.pdf", "K-IFRS_1116"),
    ("K-IFRS_제2032호_무형자산_웹사이트 원가.pdf", "K-IFRS_2032"),
)

# 산업 공통 DART 섹션 화이트리스트(모든 산업에서 유효한 재무제표 핵심 섹션).
_COMMON_SECTION_WHITELIST: tuple[str, ...] = (
    "회계정책",
    "재무상태표",
    "포괄손익계산서",
    "현금흐름표",
    "사업의 개요",
    "주요 제품",
    "매출액",
    "매출실적",
)


INDUSTRIES: dict[str, Industry] = {
    "bio": Industry(
        key="bio",
        label="제약·바이오",
        guidebook_pdfs=(("samilpwc_biology-kam.pdf", "samilpwc_biology-kam.pdf", "kam"),),
        companies=(
            TargetCompany("삼성바이오로직스", "207940"),
            TargetCompany("셀트리온", "068270"),
            TargetCompany("한미약품", "128940"),
            TargetCompany("유한양행", "000100"),
            TargetCompany("종근당", "185750"),
            TargetCompany("SK바이오팜", aliases=("에스케이바이오팜",)),
            TargetCompany("SK바이오사이언스", aliases=("에스케이바이오사이언스",)),
            TargetCompany("알테오젠"),
            TargetCompany("리가켐바이오", aliases=("레고켐바이오",)),
            TargetCompany("휴젤"),
            TargetCompany("GC녹십자", aliases=("녹십자",)),
            TargetCompany("대웅제약"),
            TargetCompany("보령", aliases=("보령제약",)),
            TargetCompany("HK이노엔"),
            TargetCompany("동아에스티"),
            TargetCompany("한올바이오파마"),
            TargetCompany("JW중외제약"),
            TargetCompany("동국제약"),
            TargetCompany("삼천당제약"),
            TargetCompany("메디톡스"),
            TargetCompany("에스티팜"),
            TargetCompany("차바이오텍"),
            TargetCompany("대원제약"),
            TargetCompany("부광약품"),
            TargetCompany("한국유나이티드제약", aliases=("유나이티드제약",)),
            TargetCompany("한독"),
            TargetCompany("안국약품"),
            TargetCompany("삼진제약"),
            TargetCompany("제일약품"),
            TargetCompany("일동제약"),
            TargetCompany("신풍제약"),
            TargetCompany("오스코텍"),
        ),
        accounting_topics=(
            "연구개발비 자산화",
            "무형자산 인식 및 상각",
            "수익인식 회계정책",
            "공정가치 평가",
        ),
        topic_keywords={
            "연구개발비 자산화": ("연구개발", "개발비", "자산화", "자본화", "경상연구개발비"),
            "무형자산 인식 및 상각": ("무형자산", "상각", "내용연수", "손상", "기술자산"),
            "수익인식 회계정책": ("수익인식", "매출", "계약", "수행의무", "거래가격"),
            "공정가치 평가": ("공정가치", "평가기법", "서열체계", "수준1", "수준2", "수준3"),
        },
        section_whitelist=_COMMON_SECTION_WHITELIST + ("무형자산", "연구개발"),
    ),
    "construction": Industry(
        key="construction",
        label="건설",
        guidebook_pdfs=(("samilpwc_construction-kam.pdf", "samilpwc_construction-kam.pdf", "kam"),),
        companies=(
            TargetCompany("삼성물산"),
            TargetCompany("현대건설"),
            TargetCompany("GS건설"),
            TargetCompany("DL이앤씨"),
            TargetCompany("대우건설"),
            # DART 등록명이 "삼성엔지니어링"에서 "삼성E&A"로 변경됨(종목코드 028050은 동일).
            # 옛 이름도 질의 감지에서 인식되도록 별칭으로 유지.
            TargetCompany("삼성E&A", "028050", aliases=("삼성엔지니어링",)),
            TargetCompany("HDC현대산업개발"),
            TargetCompany("태영건설"),
            TargetCompany("코오롱글로벌"),
            TargetCompany("계룡건설산업"),
            TargetCompany("한신공영"),
            TargetCompany("동부건설"),
        ),
        accounting_topics=(
            "공사수익인식(진행기준)",
            "미청구공사·초과청구공사",
            "하자보수충당부채",
            "PF 우발채무 및 지급보증",
        ),
        topic_keywords={
            "공사수익인식(진행기준)": ("진행률", "진행기준", "공사수익", "수행의무", "건설계약"),
            "미청구공사·초과청구공사": ("미청구공사", "초과청구공사", "공사미수금"),
            "하자보수충당부채": ("하자보수", "충당부채", "보증"),
            "PF 우발채무 및 지급보증": ("PF", "프로젝트파이낸싱", "지급보증", "우발채무", "약정"),
        },
        section_whitelist=_COMMON_SECTION_WHITELIST + ("공사수익", "수주", "우발채무", "지급보증"),
    ),
    "energy_utility": Industry(
        key="energy_utility",
        label="에너지·유틸리티",
        guidebook_pdfs=(("samilpwc_energy-utility-guidebook.pdf", "samilpwc_energy-utility-guidebook.pdf", "guidebook"),),
        companies=(
            TargetCompany("한국전력공사"),
            TargetCompany("한국가스공사"),
            TargetCompany("SK이노베이션"),
            TargetCompany("GS"),
            TargetCompany("S-Oil"),
            TargetCompany("두산에너빌리티"),
            TargetCompany("삼천리"),
            TargetCompany("한국지역난방공사"),
            TargetCompany("SK가스"),
            TargetCompany("대성에너지"),
            # DART 정식 등록명은 "서울도시가스". "서울가스"는 흔히 쓰이는 약칭이라 별칭으로 유지.
            TargetCompany("서울도시가스", "017390", aliases=("서울가스",)),
            TargetCompany("경동도시가스"),
        ),
        accounting_topics=(
            "유형자산 감가상각 및 손상",
            "정부보조금 및 배출권",
            "규제자산부채(요금규제)",
            "장기공급계약",
        ),
        topic_keywords={
            "유형자산 감가상각 및 손상": ("유형자산", "감가상각", "내용연수", "손상"),
            "정부보조금 및 배출권": ("정부보조금", "배출권", "탄소배출권"),
            "규제자산부채(요금규제)": ("규제자산", "규제부채", "요금규제"),
            "장기공급계약": ("장기공급계약", "공급계약", "취급계약"),
        },
        section_whitelist=_COMMON_SECTION_WHITELIST + ("유형자산", "정부보조금", "배출권"),
    ),
    "fnb": Industry(
        key="fnb",
        label="외식·식음료",
        guidebook_pdfs=(("samilpwc_fnb-accounting-tax-finance-guidebook.pdf", "samilpwc_fnb-accounting-tax-finance-guidebook.pdf", "guidebook"),),
        companies=(
            TargetCompany("CJ제일제당"),
            TargetCompany("오뚜기"),
            TargetCompany("농심"),
            TargetCompany("오리온"),
            TargetCompany("롯데웰푸드"),
            TargetCompany("하이트진로"),
            TargetCompany("대상"),
            TargetCompany("SPC삼립"),
            TargetCompany("매일유업"),
            TargetCompany("남양유업"),
            TargetCompany("풀무원"),
            TargetCompany("동원F&B"),
        ),
        accounting_topics=(
            "재고자산평가",
            "가맹점(프랜차이즈) 수익인식",
            "유형자산손상(매장)",
            "원재료가격변동충당부채",
        ),
        topic_keywords={
            "재고자산평가": ("재고자산", "평가손실", "저가법"),
            "가맹점(프랜차이즈) 수익인식": ("가맹점", "프랜차이즈", "로열티", "수행의무"),
            "유형자산손상(매장)": ("손상차손", "현금창출단위"),
            "원재료가격변동충당부채": ("원재료", "원가", "가격변동"),
        },
        section_whitelist=_COMMON_SECTION_WHITELIST + ("재고자산", "가맹점"),
    ),
    "b2c": Industry(
        key="b2c",
        label="소비재·유통(B2C)",
        guidebook_pdfs=(("samilpwc_b2c-accounting-guidebook.pdf", "samilpwc_b2c-accounting-guidebook.pdf", "guidebook"),),
        companies=(
            TargetCompany("롯데쇼핑"),
            TargetCompany("신세계"),
            TargetCompany("이마트"),
            TargetCompany("현대백화점"),
            TargetCompany("GS리테일"),
            TargetCompany("BGF리테일"),
            TargetCompany("LG생활건강"),
            TargetCompany("아모레퍼시픽"),
            TargetCompany("F&F"),
            TargetCompany("한섬"),
            TargetCompany("코스맥스"),
        ),
        accounting_topics=(
            "수익인식(포인트·리베이트)",
            "재고자산평가",
            "무형자산(브랜드·상표권)",
            "리스(매장임차)",
        ),
        topic_keywords={
            "수익인식(포인트·리베이트)": ("포인트", "마일리지", "리베이트", "수행의무"),
            "재고자산평가": ("재고자산", "평가손실"),
            "무형자산(브랜드·상표권)": ("상표권", "브랜드", "무형자산"),
            "리스(매장임차)": ("리스", "사용권자산", "리스부채"),
        },
        section_whitelist=_COMMON_SECTION_WHITELIST + ("무형자산", "리스"),
    ),
}


# ── 회사/산업 조회 헬퍼 ──────────────────────────────────────────────

# canonical company_name -> industry key
COMPANY_TO_INDUSTRY: dict[str, str] = {
    company.company_name: industry.key
    for industry in INDUSTRIES.values()
    for company in industry.companies
}

# 별칭/축약명 -> canonical company_name (industries.py 안의 TargetCompany.aliases 취합)
COMPANY_ALIAS_MAP: dict[str, str] = {
    alias: company.company_name
    for industry in INDUSTRIES.values()
    for company in industry.companies
    for alias in company.aliases
}

# 회사명 표기 길이 내림차순(긴 이름부터 매칭해야 "GS"가 "GS건설"을 오매칭하지 않음)
_ALL_COMPANY_SURFACE_FORMS: list[str] = sorted(
    {*COMPANY_TO_INDUSTRY.keys(), *COMPANY_ALIAS_MAP.keys()}, key=len, reverse=True
)

NEGATION_MARKERS: tuple[str, ...] = ("아닌", "말고", "제외", "빼고", "이외", "외의", "외에", "아니라", "아니고")

# 질의에 회사명이 없어도 산업 자체를 직접 언급하는 경우를 잡기 위한 키워드.
# [주의] "건설업"처럼 접미사가 고정된 키워드만 쓰면 "건설 기업"/"건설 회사"처럼 띄어쓰기가
# 들어간 자연스러운 표현을 놓친다(2026-07-25 진단: "에너지 기업"이 "에너지업"과 매칭 안 돼
# 산업 필터링이 통째로 빠지면서 엉뚱한 산업의 가이드북/사례가 섞여 나온 버그).
# 그래서 짧은 어근(語根) 위주로 넓게 잡는다 — 어근은 "~업/~회사/~기업/~산업/~분야" 등
# 어떤 접미사가 붙어도 부분 문자열로 항상 걸린다.
INDUSTRY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "bio": ("제약", "바이오", "신약", "임상"),
    "construction": ("건설", "시공"),
    "energy_utility": ("에너지", "유틸리티", "전력", "가스"),
    "fnb": ("외식", "식품", "식음료", "프랜차이즈"),
    "b2c": ("유통", "소비재", "리테일"),
}


def _canonical_company(name: str) -> str:
    return COMPANY_ALIAS_MAP.get(name, name)


def detect_industries_and_companies(query: str) -> tuple[set[str], list[str], list[str]]:
    """질의에서 산업/기업을 감지한다.

    Returns: (감지된 industry key 집합, 포함 대상 기업(정식명), 제외 대상 기업(정식명))
    회사명이 매칭되면 해당 회사의 소속 산업도 자동으로 감지 집합에 들어간다.
    회사명 없이 산업 키워드만 언급된 경우도 industries에 반영된다.
    """
    positive: list[str] = []
    negated: list[str] = []
    industries: set[str] = set()

    # 긴 이름부터 매칭하되(_ALL_COMPANY_SURFACE_FORMS가 길이 내림차순), 이미 긴 이름이 차지한
    # 문자 구간은 "claimed"로 표시해 그 안에 포함된 짧은 이름이 별개로 재매칭되지 않게 한다.
    # (예: "GS건설"이 매칭되면, 그 안에 포함된 "GS"(별도 회사, 에너지업종)는 건너뛴다 —
    # 그렇지 않으면 "GS건설"만 언급했는데 "GS"의 소속 산업(energy_utility)까지 오검출된다.)
    claimed: list[tuple[int, int]] = []

    def _is_claimed(start: int, end: int) -> bool:
        return any(start < c_end and end > c_start for c_start, c_end in claimed)

    for surface in _ALL_COMPANY_SURFACE_FORMS:
        idx = query.find(surface)
        if idx == -1:
            continue
        end = idx + len(surface)
        if _is_claimed(idx, end):
            continue
        claimed.append((idx, end))
        canonical = _canonical_company(surface)
        window = query[end: end + 8]
        if any(m in window for m in NEGATION_MARKERS):
            negated.append(canonical)
        else:
            positive.append(canonical)

    positive = [c for c in dict.fromkeys(positive) if c not in negated]
    negated = list(dict.fromkeys(negated))

    for company in positive:
        ind = COMPANY_TO_INDUSTRY.get(company)
        if ind:
            industries.add(ind)

    for ind_key, keywords in INDUSTRY_KEYWORDS.items():
        if any(kw in query for kw in keywords):
            industries.add(ind_key)

    return industries, positive, negated


def all_companies_flat() -> list[TargetCompany]:
    """모든 산업의 TargetCompany를 하나의 리스트로 (DART 전역 회사명 감지용)."""
    return [c for industry in INDUSTRIES.values() for c in industry.companies]
