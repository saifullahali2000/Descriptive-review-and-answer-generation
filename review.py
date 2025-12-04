import io
import json
import re
import time
from typing import Dict, Any, List, Optional

import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cheat Sheet Checker + Answers", layout="wide")
st.title("📋 Questions vs Cheat Sheet (Alignment + Grammar + Answers)")
st.caption("Outputs persist (won’t vanish on dropdown changes). Answers are plain text (no markdown fences).")

# ---------------- SESSION STATE ----------------
if "res_df" not in st.session_state:
    st.session_state.res_df = None
if "answers_df" not in st.session_state:
    st.session_state.answers_df = None
if "ran_once" not in st.session_state:
    st.session_state.ran_once = False


# ---------------- NORMALIZE ----------------
def norm(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\ufeff", "").replace("\u200b", "").replace("\u00a0", " ")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


# ---------------- READERS ----------------
def read_delimited_text(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    encodings = ["utf-8", "utf-8-sig", "utf-16", "cp1252", "latin1"]
    seps = [",", ";", "\t", "|"]
    last_err = None

    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(
                    io.BytesIO(raw),
                    encoding=enc,
                    sep=sep,
                    engine="python",
                    on_bad_lines="skip",
                )
                if df.shape[1] < 2:
                    continue
                return df
            except Exception as e:
                last_err = e

    raise last_err if last_err else ValueError("Could not read delimited file.")


def list_excel_sheets(uploaded_file) -> List[str]:
    uploaded_file.seek(0)
    xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
    return xls.sheet_names


def read_excel_sheet(uploaded_file, sheet_name: str) -> pd.DataFrame:
    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file, engine="openpyxl", sheet_name=sheet_name)


def read_table_file(uploaded_file, sheet_name: Optional[str] = None) -> pd.DataFrame:
    name = (uploaded_file.name or "").lower()
    if name.endswith((".xlsx", ".xls")):
        if not sheet_name:
            sheet_name = list_excel_sheets(uploaded_file)[0]
        return read_excel_sheet(uploaded_file, sheet_name)
    return read_delimited_text(uploaded_file)


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df is None or df.shape[1] == 0:
        return df

    df.columns = [norm(c) for c in df.columns]

    q_candidates = {"question text", "question", "questions", "question_text", "questiontext"}
    m_candidates = {"mark", "marks", "max marks", "maxmarks", "max_mark", "score"}

    q_col = None
    m_col = None

    for c in df.columns:
        if c in q_candidates:
            q_col = c
        if c in m_candidates:
            m_col = c

    if q_col is None:
        for c in df.columns:
            if "question" in c:
                q_col = c
                break

    if m_col is None:
        for c in df.columns:
            if "mark" in c or "score" in c:
                m_col = c
                break

    if (q_col is None or m_col is None) and df.shape[1] >= 2:
        q_col = q_col or df.columns[0]
        m_col = m_col or df.columns[1]

    if q_col is not None:
        df = df.rename(columns={q_col: "Question Text"})
    if m_col is not None:
        df = df.rename(columns={m_col: "mark"})

    return df


def validate_df(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.shape[1] == 0:
        return "No readable data found in file."
    needed = {"Question Text", "mark"}
    missing = needed - set(df.columns)
    if missing:
        return f"File is missing columns: {', '.join(missing)}"
    if df["Question Text"].isna().all():
        return "All 'Question Text' values are empty."
    return None


def normalize_question_text(q: str) -> str:
    q = "" if pd.isna(q) else str(q)
    return q.replace("\\n", "\n").strip()


def parse_mark(m) -> int:
    try:
        if pd.isna(m):
            return 1
        if isinstance(m, str):
            m = m.strip()
        val = int(float(m))
        return max(1, val)
    except Exception:
        return 1


# ---------------- ANSWER CLEANUP: remove accidental markdown fences ----------------
def strip_markdown_fences(text: str) -> str:
    if not text:
        return ""
    # Convert ```...\n...\n``` blocks to indented code
    def repl(match):
        code = (match.group(2) or "").strip("\n")
        lines = code.splitlines()
        return "\n" + "\n".join("    " + ln for ln in lines) + "\n"

    text = re.sub(r"```(\w+)?\n(.*?)\n```", repl, text, flags=re.DOTALL)
    text = text.replace("```", "").replace("`", "")
    return text.strip()


# ---------------- LLM CALLS ----------------
def call_openai(api_key: str, model: str, messages: List[Dict[str, str]], temperature: float) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return resp.choices[0].message.content or ""


def call_perplexity(api_key: str, model: str, messages: List[Dict[str, str]], temperature: float) -> str:
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature}
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    if r.status_code >= 400:
        raise RuntimeError(f"Perplexity API error {r.status_code}: {r.text}")
    data = r.json()
    return (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""


def llm_json(provider: str, api_key: str, model: str, messages: List[Dict[str, str]], temperature: float) -> Dict[str, Any]:
    for attempt in range(2):
        text = call_openai(api_key, model, messages, temperature) if provider == "OpenAI" else call_perplexity(api_key, model, messages, temperature)
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
        if attempt == 0:
            messages = messages + [{"role": "user", "content": "Return ONLY valid JSON. No extra text."}]
    raise ValueError("Could not parse JSON.")


def analyze_batch(provider, api_key, model, cheat_sheet_md, questions, strictness, temperature):
    cs = cheat_sheet_md.strip()
    if len(cs) > 24000:
        cs = cs[:24000] + "\n\n[CHEAT SHEET TRUNCATED]"

    system = f"""
You are a strict syllabus-alignment and question-quality reviewer AND answer writer.

OUTPUT REQUIREMENT:
- Answer must be PLAIN TEXT ONLY.
- Do NOT use triple backticks or markdown.
- If including code, show as indented lines (4 spaces).
- Use clean formatting: Definition:, Explanation:, Example:, and '-' bullet points.

Rules:
- If not aligned with cheat sheet: is_aligned='no' and answer=''.
- Do not hallucinate outside cheat sheet.
- Do not change code logic.

Answer depth by marks:
1: 2–4 lines
2: short paragraph + 2 bullets
4: structured + example
8: detailed step-by-step + example + key points

Return ONLY JSON:
{{
  "results": [
    {{
      "question_text": "...",
      "mark": 1,
      "is_aligned": "yes|no",
      "alignment_reason": "short 1-2 sentences",
      "grammar_score": 1-10,
      "has_issues": "yes|no",
      "issues": ["..."],
      "improved_question": "",
      "cheatsheet_evidence": ["..."],
      "answer": ""
    }}
  ]
}}
Strictness={strictness}
""".strip()

    user = {"role": "user", "content": json.dumps({"cheat_sheet_markdown": cs, "questions": questions}, ensure_ascii=False)}
    out = llm_json(provider, api_key, model, [{"role": "system", "content": system}, user], temperature)

    results = out.get("results", [])
    for r in results:
        r["answer"] = strip_markdown_fences(str(r.get("answer", "")).strip())
        r["improved_question"] = strip_markdown_fences(str(r.get("improved_question", "")).strip())
    return results


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("🔐 LLM Settings")
    provider = st.selectbox("Provider", ["OpenAI", "Perplexity"])
    api_key = st.text_input("Paste API Key", type="password")

    st.subheader("Model")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"] if provider == "OpenAI" else ["sonar-pro", "sonar"], index=0)

    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    strictness = st.selectbox("Alignment strictness", ["strict", "medium", "lenient"], index=0)
    batch_size = st.slider("Questions per call", 3, 20, 6, 1)
    sleep_s = st.slider("Delay between calls (sec)", 0.0, 2.0, 0.1, 0.1)


# ---------------- INPUTS ----------------
st.subheader("1) Paste Cheat Sheet (Markdown)")
cheat_sheet_md = st.text_area("Paste your cheat sheet here.", height=260)

st.subheader("2) Upload Questions File")
uploaded = st.file_uploader("Upload .xlsx/.xls or .csv/.tsv", type=["xlsx", "xls", "csv", "tsv"])

sheet = None
if uploaded and uploaded.name.lower().endswith((".xlsx", ".xls")):
    try:
        sheet = st.selectbox("Select Excel sheet", list_excel_sheets(uploaded), index=0)
    except Exception as e:
        st.error(f"Could not read Excel sheet names: {e}")

# Button triggers analysis and stores outputs
if st.button("🚀 Analyze + Generate Answers", disabled=(not uploaded or not cheat_sheet_md.strip() or not api_key)):
    try:
        df_raw = read_table_file(uploaded, sheet_name=sheet)
        df = map_columns(df_raw)
        err = validate_df(df)
        if err:
            st.error(err)
            st.stop()

        df = df.copy()
        df["Question Text"] = df["Question Text"].map(normalize_question_text)
        df["mark"] = df["mark"].map(parse_mark)

        payload_questions = [{"question_text": qt, "mark": mk} for qt, mk in zip(df["Question Text"], df["mark"])]

        st.info(f"Loaded {len(payload_questions)} questions. Running analysis...")

        all_results: List[Dict[str, Any]] = []
        progress = st.progress(0)
        status = st.empty()
        total = len(payload_questions)

        for i in range(0, total, batch_size):
            batch = payload_questions[i:i + batch_size]
            status.write(f"Analyzing {i+1} → {min(i+batch_size, total)}")
            all_results.extend(analyze_batch(provider, api_key, model, cheat_sheet_md, batch, strictness, temperature))
            progress.progress(min((i + len(batch)) / total, 1.0))
            time.sleep(sleep_s)

        res_df = pd.DataFrame(all_results).fillna("")
        answers_df = res_df[["question_text", "mark", "is_aligned", "answer"]].copy()

        st.session_state.res_df = res_df
        st.session_state.answers_df = answers_df
        st.session_state.ran_once = True

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# ---------------- RENDER STORED RESULTS (PERSISTS ON DROPDOWN CHANGE) ----------------
if st.session_state.res_df is not None and st.session_state.answers_df is not None:
    res_df = st.session_state.res_df
    answers_df = st.session_state.answers_df

    st.subheader("✅ Answers (Separate Table)")
    st.dataframe(answers_df, use_container_width=True)

    st.subheader("🔎 Answer Viewer (Plain Text)")
    selected_q = st.selectbox("Select a question to view its answer", answers_df["question_text"].tolist())
    row = answers_df[answers_df["question_text"] == selected_q].iloc[0]
    st.write(f"Marks: {row['mark']} | Aligned: {row['is_aligned']}")
    if str(row["is_aligned"]).lower() == "yes" and str(row["answer"]).strip():
        st.text(row["answer"])
    else:
        st.warning("Not aligned with cheat sheet, so no answer was generated.")

    st.subheader("✍️ Questions with Issues (Original → Improved)")
    issues_df = res_df[res_df["has_issues"].astype(str).str.lower().eq("yes")].copy()
    if issues_df.empty:
        st.success("No grammar/formation issues found.")
    else:
        st.dataframe(issues_df[["question_text", "mark", "grammar_score", "issues", "improved_question"]], use_container_width=True)

    st.subheader("❌ Not Aligned with Cheat Sheet")
    not_aligned_df = res_df[res_df["is_aligned"].astype(str).str.lower().eq("no")].copy()
    if not not_aligned_df.empty:
        st.dataframe(not_aligned_df[["question_text", "mark", "alignment_reason", "cheatsheet_evidence"]], use_container_width=True)
    else:
        st.success("All questions appear aligned (based on selected strictness).")

    st.subheader("⬇️ Download")
    st.download_button("Download Full Results CSV", data=res_df.to_csv(index=False).encode("utf-8"),
                       file_name="analysis_results_with_plaintext_answers.csv", mime="text/csv")
    st.download_button("Download Answers CSV", data=answers_df.to_csv(index=False).encode("utf-8"),
                       file_name="answers_plain_text.csv", mime="text/csv")

else:
    if st.session_state.ran_once:
        st.warning("No stored results found. Click 'Analyze + Generate Answers' again.")
