/* app.py의 render_answer_with_inline_sources / _build_cards_html / _trim_to_complete_sentences
   로직을 그대로 JS로 옮긴 것 — 새 디자인이 아니라 같은 렌더링 규칙의 이식이다. */

const chatLog = document.getElementById("chat-log");
const input = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");

// 대화 기록을 localStorage에 영속화하지 않는다 — 새로고침/재방문할 때마다 항상 빈 채팅으로
// 시작한다(사용자 요청). messages는 이 페이지 로드 동안만 메모리에 유지되는 배열이다.
localStorage.removeItem("auditgpt_messages"); // 예전 버전이 저장해둔 기록 정리
let messages = [];

const EXAMPLE_QUESTIONS = [
  "K-IFRS 제1109호 금융상품은 어떻게 분류해?",
  "셀트리온의 최근 연구개발비 자산화 현황을 알려줘",
  "GS건설의 공사수익인식 방법과 최근 재무 수치를 알려줘",
  "삼성바이오로직스와 셀트리온의 연구개발비 처리 방식을 비교해줘",
  "건설업 KAM에서 자주 언급되는 감사 위험은 뭐야?",
  "지금 어떤 산업, 어떤 기업 데이터를 갖고 있어?",
];

const CATEGORY_LABELS = {
  kifrs: "📖 기준서 (K-IFRS)",
  guidebook: "📗 가이드북 · KAM",
  dart: "📊 DART 공시",
};
const CATEGORY_ORDER = ["kifrs", "guidebook", "dart"];

function escapeHtml(s) {
  return (s || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// app.py:_trim_to_complete_sentences 포트 (500자 제한, 문장 경계 인식)
function trimToCompleteSentences(text, maxChars = 500) {
  text = (text || "").trim();
  if (text.length <= maxChars) return text;
  const chunk = text.slice(0, maxChars);
  const sentenceEndRe = /[다요임음.!?。]\s/g;
  let lastEnd = -1;
  let m;
  while ((m = sentenceEndRe.exec(chunk)) !== null) {
    lastEnd = m.index + m[0].length;
  }
  if (lastEnd !== -1) {
    return chunk.slice(0, lastEnd).trimEnd();
  }
  const lastPeriod = chunk.lastIndexOf(".");
  if (lastPeriod > chunk.length / 4) {
    return chunk.slice(0, lastPeriod + 1);
  }
  return chunk;
}

// app.py:_build_cards_html 포트
function buildCardsHtml(sids, catalogMap) {
  let cards = "";
  sids.forEach((sid) => {
    const item = catalogMap[sid];
    if (!item) return;
    const source = item.source || "";
    const company = item.company || "";
    const section = item.section || "";
    const page = item.page || "";
    const content = trimToCompleteSentences(item.content || "");
    const contentEscaped = escapeHtml(content);
    const isDart = source.toUpperCase().startsWith("DART");
    const labelParts = [source];
    if (company && !isDart) labelParts.push(company);
    if (section) labelParts.push(section);
    if (page && !isDart) labelParts.push(`p.${page}`);
    const label = labelParts.join(" | ");
    cards += `<div class="src-card"><div class="src-label">${label}</div><div class="src-tooltip">${contentEscaped}</div></div>`;
  });
  return cards ? `<div class="src-cards">${cards}</div>` : "";
}

// 카테고리(기준서/가이드북·KAM/DART)별로 묶어, 답변 본문에서 [S#]로 인용되지 않았더라도
// 실제 검색에 쓰인 출처는 항상 보이도록 하는 전체 참고 자료 섹션.
function buildReferenceSectionHtml(sources) {
  if (!sources || !sources.length) return "";
  const byCategory = {};
  sources.forEach((item) => {
    const cat = item.category || "기타";
    (byCategory[cat] = byCategory[cat] || []).push(item);
  });
  const order = [...CATEGORY_ORDER, ...Object.keys(byCategory).filter((c) => !CATEGORY_ORDER.includes(c))];
  let body = "";
  order.forEach((cat) => {
    const items = byCategory[cat];
    if (!items || !items.length) return;
    const label = CATEGORY_LABELS[cat] || cat;
    const catalogMap = Object.fromEntries(items.map((it) => [it.sid, it]));
    const cardsHtml = buildCardsHtml(items.map((it) => it.sid), catalogMap);
    if (cardsHtml) {
      body += `<div class="ref-group-label">${label}</div>${cardsHtml}`;
    }
  });
  return body ? `<div class="ref-section">${body}</div>` : "";
}

// app.py:render_answer_with_inline_sources 포트 + 전체 참고 자료 섹션 추가
function renderAnswerWithSources(container, answer, sources) {
  const catalogMap = {};
  (sources || []).forEach((item) => {
    catalogMap[item.sid] = item;
  });

  const paragraphs = (answer || "").split(/\n\n/);
  paragraphs.forEach((para) => {
    if (!para.trim()) return;
    const sids = [...new Set([...para.matchAll(/\[(S\d+)\]/g)].map((m) => m[1]))];
    const cleanPara = para.replace(/\s*\[(S\d+)\]/g, "").trim();
    if (!cleanPara) return;

    const p = document.createElement("div");
    p.innerHTML = marked.parse(cleanPara);
    container.appendChild(p);

    if (sids.length) {
      const cardsHtml = buildCardsHtml(sids, catalogMap);
      if (cardsHtml) {
        const wrap = document.createElement("div");
        wrap.innerHTML = cardsHtml;
        container.appendChild(wrap.firstElementChild);
      }
    }
  });

  const refHtml = buildReferenceSectionHtml(sources);
  if (refHtml) {
    const wrap = document.createElement("div");
    wrap.innerHTML = refHtml;
    container.appendChild(wrap.firstElementChild);
  }
}

function appendUserMessage(text) {
  const row = document.createElement("div");
  row.className = "chat-message user";
  const content = document.createElement("div");
  content.className = "chat-content";
  const p = document.createElement("p");
  p.textContent = text;
  content.appendChild(p);
  row.appendChild(content);
  chatLog.appendChild(row);
}

function appendAssistantContainer() {
  const row = document.createElement("div");
  row.className = "chat-message assistant";
  const content = document.createElement("div");
  content.className = "chat-content";
  row.appendChild(content);
  chatLog.appendChild(row);
  return content;
}

function removeSuggestions() {
  chatLog.querySelectorAll(".suggestions").forEach((el) => el.remove());
}

// 채팅 시작 전(빈 상태)과 답변 직후 모두에서 예시 질문 칩을 보여준다.
function appendSuggestions() {
  removeSuggestions();
  const wrap = document.createElement("div");
  wrap.className = "suggestions";
  const label = document.createElement("div");
  label.className = "suggestions-label";
  label.textContent = "이런 질문은 어떠세요?";
  wrap.appendChild(label);
  EXAMPLE_QUESTIONS.forEach((q) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "suggestion-chip";
    chip.textContent = q;
    chip.addEventListener("click", () => {
      input.value = q;
      sendQuery();
    });
    wrap.appendChild(chip);
  });
  chatLog.appendChild(wrap);
}

function renderHistory() {
  chatLog.innerHTML = "";
  messages.forEach((m) => {
    if (m.role === "user") {
      appendUserMessage(m.content);
    } else {
      const content = appendAssistantContainer();
      renderAnswerWithSources(content, m.content, m.sources || []);
    }
  });
  appendSuggestions();
}

async function sendQuery() {
  const query = input.value.trim();
  if (!query) return;
  input.value = "";
  sendBtn.disabled = true;
  removeSuggestions();

  appendUserMessage(query);
  messages.push({ role: "user", content: query });

  const assistantContent = appendAssistantContainer();
  assistantContent.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        history: messages.slice(0, -1).map((m) => ({ role: m.role, content: m.content })),
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    assistantContent.innerHTML = "";
    renderAnswerWithSources(assistantContent, data.answer, data.sources || []);
    messages.push({ role: "assistant", content: data.answer, sources: data.sources || [] });
  } catch (e) {
    assistantContent.innerHTML = `<p>오류가 발생했습니다: ${escapeHtml(String(e.message || e))}</p>`;
  }

  appendSuggestions();
}

input.addEventListener("input", () => {
  sendBtn.disabled = input.value.trim().length === 0;
});
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendQuery();
  }
});
sendBtn.addEventListener("click", sendQuery);

renderHistory();
