import os
import re
import json
import sqlite3
import hashlib
from datetime import datetime

import streamlit as st
import fitz
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------- CONFIG ----------------------------
load_dotenv()
MODEL = "gemini-3.8-flash"
DB_PATH = "noteproof_history.db"

st.set_page_config(
    page_title="NoteProof | AI Trust Workspace",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------- STYLE -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:#0b1020;color:#edf2ff;}
[data-testid="stHeader"] {background:rgba(11,16,32,.92);}
[data-testid="stSidebar"] {background:#10182b;border-right:1px solid #26324b;}
.block-container {padding-top:2rem;padding-bottom:3rem;max-width:1450px;}
.hero {padding:28px 30px;border:1px solid #283653;border-radius:22px;
background:linear-gradient(120deg,#172743 0%,#172039 55%,#211d43 100%);margin-bottom:22px;}
.eyebrow {color:#91a9d8;text-transform:uppercase;letter-spacing:2px;font-size:11px;font-weight:700;}
.hero h1 {font-family:Manrope,sans-serif;font-size:35px;line-height:1.15;margin:9px 0;color:#fff;}
.hero p {color:#b8c5df;font-size:15px;margin:0;max-width:760px;}
.panel {background:#121c31;border:1px solid #293754;border-radius:17px;padding:20px 22px;margin-bottom:16px;}
.metric {background:linear-gradient(145deg,#172743,#131d32);border:1px solid #2b3a59;border-radius:16px;padding:18px;}
.metric-label {color:#9eafd0;font-size:12px;text-transform:uppercase;letter-spacing:1px;}
.metric-value {font-family:Manrope,sans-serif;font-size:30px;font-weight:800;color:#fff;margin-top:6px;}
.metric-note {color:#8fa2c5;font-size:12px;margin-top:2px;}
.small-muted {color:#91a1bf;font-size:12px;}
.verdict-supported {background:#103a34;color:#76e3c1;border:1px solid #1d725f;padding:7px 12px;border-radius:999px;font-weight:700;font-size:12px;}
.verdict-contradicted {background:#421f2c;color:#ff9cab;border:1px solid #8c3d50;padding:7px 12px;border-radius:999px;font-weight:700;font-size:12px;}
.verdict-unverifiable {background:#3b321b;color:#f4d27e;border:1px solid #80692e;padding:7px 12px;border-radius:999px;font-weight:700;font-size:12px;}
div.stButton > button {border-radius:10px;border:1px solid #4969a8;background:#4778e8;color:white;font-weight:700;padding:.65rem 1.1rem;}
div.stButton > button:hover {border-color:#8caeff;background:#3567d7;color:white;}
.stTextArea textarea,.stTextInput input {background:#101a2d;color:#edf2ff;border:1px solid #33435f;border-radius:12px;}
[data-testid="stFileUploader"] {background:#111c30;border:1px dashed #52688f;border-radius:14px;padding:12px;}
hr {border-color:#293754;}
h1,h2,h3 {font-family:Manrope,sans-serif;}
</style>
""", unsafe_allow_html=True)

# ---------------------------- DATABASE --------------------------
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS checks(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT, file_names TEXT, total INTEGER,
        supported INTEGER, contradicted INTEGER, unverifiable INTEGER,
        result_json TEXT
    )""")
    con.commit()
    con.close()

def save_check(file_names, results):
    counts = {k: sum(1 for r in results if r.get("verdict") == k)
              for k in ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]}
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO checks(created_at,file_names,total,supported,contradicted,unverifiable,result_json) VALUES(?,?,?,?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M"), ", ".join(file_names),
         len(results), counts["SUPPORTED"], counts["CONTRADICTED"],
         counts["UNVERIFIABLE"], json.dumps(results, ensure_ascii=False))
    )
    con.commit()
    con.close()

def load_history():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT id,created_at,file_names,total,supported,contradicted,unverifiable FROM checks ORDER BY id DESC",
        con
    )
    con.close()
    return df

init_db()

# ---------------------------- PDF / RETRIEVAL -------------------
def extract_pdf_pages(uploaded_files):
    pages = []
    for file in uploaded_files:
        try:
            doc = fitz.open(stream=file.getvalue(), filetype="pdf")
            for page_index, page in enumerate(doc):
                text = page.get_text("text").strip()
                if text:
                    # Keep chunks short enough for precise evidence references.
                    words = text.split()
                    chunk_size = 180
                    for start in range(0, len(words), chunk_size):
                        chunk = " ".join(words[start:start + chunk_size])
                        if len(chunk) > 35:
                            pages.append({
                                "source": file.name,
                                "page": page_index + 1,
                                "text": chunk
                            })
            doc.close()
        except Exception as exc:
            st.warning(f"Could not read {file.name}: {exc}")
    return pages

def retrieve_evidence(claim, chunks, top_k=3):
    if not chunks:
        return []
    corpus = [c["text"] for c in chunks]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(corpus + [claim])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
        ranked = scores.argsort()[::-1][:top_k]
        evidence = []
        for idx in ranked:
            if scores[idx] > 0:
                item = dict(chunks[int(idx)])
                item["similarity"] = float(scores[idx])
                evidence.append(item)
        return evidence
    except ValueError:
        return []

# ---------------------------- GEMINI ----------------------------
def get_client():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")
    return genai.Client(api_key=key)

def call_gemini(prompt):
    client = get_client()
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            max_output_tokens=5000,
        ),
    )
    raw = response.text or ""
    # Handle occasional fenced JSON responses safely.
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    return json.loads(raw)

def split_into_claims(answer):
    prompt = f"""
You are an academic fact-checking assistant. Extract only factual, independently checkable
claims from the supplied AI-generated answer. Do not decide whether they are true.
Keep each claim concise and preserve its meaning. Ignore greetings, opinions and instructions.
Return ONLY valid JSON in this exact shape:
{{"claims":["claim 1","claim 2"]}}
If no factual claims exist, return {{"claims":[]}}.

ANSWER:
{answer[:14000]}
"""
    data = call_gemini(prompt)
    claims = data.get("claims", [])
    return [str(c).strip() for c in claims if str(c).strip()][:20]

def verify_claim(claim, evidence):
    if not evidence:
        return {
            "claim": claim, "verdict": "UNVERIFIABLE", "confidence": 0,
            "explanation": "No sufficiently relevant passage was retrieved from the uploaded notes.",
            "evidence": []
        }

    evidence_text = "\n\n".join(
        f"[E{i+1}] Source: {e['source']} | Page: {e['page']} | Passage: {e['text']}"
        for i, e in enumerate(evidence)
    )
    prompt = f"""
You are a careful evidence-grounded academic verifier.
Judge the claim ONLY against the supplied passages. The uploaded notes are the evidence boundary.
Do not use outside knowledge. A related passage is not automatically proof.
SUPPORTED = the passages clearly support the claim.
CONTRADICTED = the passages clearly state something incompatible with the claim.
UNVERIFIABLE = evidence is absent, ambiguous, incomplete, or only loosely related.
Use UNVERIFIABLE rather than guessing. Confidence is your confidence in the verdict from 0 to 100.
Return ONLY valid JSON:
{{
 "verdict":"SUPPORTED or CONTRADICTED or UNVERIFIABLE",
 "confidence":0,
 "explanation":"brief reason grounded in the passages",
 "evidence_ids":["E1"]
}}

CLAIM:
{claim}

RETRIEVED PASSAGES:
{evidence_text}
"""
    data = call_gemini(prompt)
    verdict = str(data.get("verdict", "UNVERIFIABLE")).upper()
    if verdict not in {"SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"}:
        verdict = "UNVERIFIABLE"
    selected_ids = set(data.get("evidence_ids", []))
    selected = []
    for i, item in enumerate(evidence):
        if f"E{i+1}" in selected_ids:
            selected.append(item)
    if not selected:
        selected = evidence[:1]
    confidence = max(0, min(100, int(data.get("confidence", 0))))
    # A low-confidence result must abstain instead of forcing a verdict.
    if confidence < 55:
        verdict = "UNVERIFIABLE"
    return {
        "claim": claim,
        "verdict": verdict,
        "confidence": confidence,
        "explanation": str(data.get("explanation", "The evidence was insufficient for a confident decision.")),
        "evidence": selected
    }

def run_verification(answer, chunks, progress=None):
    claims = split_into_claims(answer)
    results = []
    for i, claim in enumerate(claims):
        evidence = retrieve_evidence(claim, chunks)
        result = verify_claim(claim, evidence)
        results.append(result)
        if progress:
            progress.progress((i + 1) / max(1, len(claims)), text=f"Checking claim {i+1} of {len(claims)}")
    return results

# ---------------------------- UI HELPERS ------------------------
def hero(title, subtitle, tag="AI TRUST WORKSPACE"):
    st.markdown(f"""
    <div class="hero">
      <div class="eyebrow">{tag}</div>
      <h1>{title}</h1>
      <p>{subtitle}</p>
    </div>""", unsafe_allow_html=True)

def metric(label, value, note):
    st.markdown(f"""
    <div class="metric">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-note">{note}</div>
    </div>""", unsafe_allow_html=True)

def verdict_badge(verdict):
    css = {
        "SUPPORTED": "verdict-supported",
        "CONTRADICTED": "verdict-contradicted",
        "UNVERIFIABLE": "verdict-unverifiable",
    }.get(verdict, "verdict-unverifiable")
    st.markdown(f'<span class="{css}">{verdict}</span>', unsafe_allow_html=True)

def show_results(results):
    if not results:
        st.info("No factual claims were detected in the supplied answer.")
        return
    counts = {k: sum(1 for r in results if r["verdict"] == k)
              for k in ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]}
    a, b, c, d = st.columns(4)
    with a: metric("Claims checked", len(results), "Individual factual statements")
    with b: metric("Supported", counts["SUPPORTED"], "Grounded in your notes")
    with c: metric("Contradicted", counts["CONTRADICTED"], "Conflicts with your notes")
    with d: metric("Unverifiable", counts["UNVERIFIABLE"], "Insufficient evidence")

    st.markdown("### Claim-by-claim findings")
    for idx, item in enumerate(results, 1):
        with st.container(border=True):
            left, right = st.columns([5, 1])
            with left:
                st.markdown(f"**{idx:02d} · {item['claim']}**")
            with right:
                verdict_badge(item["verdict"])
            st.progress(item["confidence"] / 100, text=f"Verdict confidence: {item['confidence']}%")
            st.write(item["explanation"])
            if item["evidence"]:
                with st.expander(f"View evidence ({len(item['evidence'])} passage(s))", expanded=True):
                    for ev in item["evidence"]:
                        st.markdown(f"**{ev['source']} · Page {ev['page']}**")
                        st.write(ev["text"])
                        st.caption(f"Retrieval relevance: {ev['similarity']:.3f}")
            else:
                st.caption("No supporting passage was found in the uploaded notes.")
    report = {
        "generated_at": datetime.now().isoformat(timespec="minutes"),
        "model": MODEL,
        "results": results,
        "summary": counts
    }
    st.download_button(
        "⬇ Download verification report (JSON)",
        data=json.dumps(report, indent=2, ensure_ascii=False),
        file_name="noteproof_verification_report.json",
        mime="application/json"
    )

# ---------------------------- SIDEBAR ---------------------------
with st.sidebar:
    st.markdown("## 🔎 NoteProof")
    st.caption("Evidence-led AI answer verification")
    st.divider()
    page = st.radio(
        "WORKSPACE",
        ["Overview", "Verify an answer", "Evaluation lab", "History"],
        label_visibility="visible"
    )
    st.divider()
    st.markdown("**SYSTEM STATUS**")
    api_ready = bool(os.getenv("GEMINI_API_KEY", "").strip())
    st.markdown("🟢 Gemini key detected" if api_ready else "🔴 Gemini key missing")
    st.caption(f"Model: {MODEL}")
    st.caption("Your PDFs are processed locally for text retrieval.")
    st.divider()
    st.caption("NoteProof is an academic aid, not a guarantee of truth.")

# ---------------------------- PAGES -----------------------------
if page == "Overview":
    hero("Trust every claim. Not every confident answer.",
         "NoteProof checks AI-generated academic answers against your own lecture notes, showing the evidence behind every verdict.")
    col1, col2, col3 = st.columns(3)
    with col1: metric("Evidence-first", "01", "Retrieve passages before asking AI to judge")
    with col2: metric("Three outcomes", "3", "Supported · Contradicted · Unverifiable")
    with col3: metric("Traceability", "PDF", "Every finding links back to a file and page")
    st.markdown("### How NoteProof works")
    steps = st.columns(4)
    for box, num, title, desc in zip(
        steps, ["01", "02", "03", "04"],
        ["Upload notes", "Extract claims", "Retrieve evidence", "Verify & report"],
        ["Add lecture PDFs as your evidence base.", "AI separates an answer into checkable statements.", "Relevant page passages are retrieved for each claim.", "AI compares claim to evidence and explains its verdict."]
    ):
        with box:
            st.markdown(f'<div class="panel"><div class="eyebrow">STEP {num}</div><h3>{title}</h3><p class="small-muted">{desc}</p></div>', unsafe_allow_html=True)
    st.markdown("### Built for the AI-FEST challenge")
    st.info("Primary focus: hallucination and accountability in AI-generated academic answers. NoteProof deliberately returns UNVERIFIABLE when the uploaded notes do not provide enough evidence.")
    hist = load_history()
    if not hist.empty:
        st.markdown("### Recent activity")
        st.dataframe(hist.head(5), use_container_width=True, hide_index=True)

elif page == "Verify an answer":
    hero("Verify an AI-generated answer",
         "Upload your source notes, paste an answer, and inspect the evidence behind every individual claim.")
    uploaded = st.file_uploader("1 · Upload lecture notes (PDF)", type=["pdf"], accept_multiple_files=True)
    answer = st.text_area(
        "2 · Paste the AI-generated answer",
        height=220,
        placeholder="Paste the answer you want to verify here..."
    )
    if uploaded:
        with st.spinner("Reading PDF pages and preparing evidence index..."):
            chunks = extract_pdf_pages(uploaded)
        st.success(f"{len(uploaded)} PDF(s) loaded · {len(chunks)} searchable evidence chunks extracted.")
        with st.expander("Uploaded source files"):
            for f in uploaded:
                st.write(f"📄 {f.name}")
    else:
        chunks = []
        st.caption("Your PDFs are only used as the evidence base for this verification session.")
    run = st.button("🔎 Verify claims", type="primary", use_container_width=False)
    if run:
        if not uploaded or not chunks:
            st.error("Upload at least one readable PDF first.")
        elif not answer.strip():
            st.error("Paste an AI-generated answer first.")
        elif len(answer) > 14000:
            st.error("Please keep the answer under 14,000 characters for this demo.")
        else:
            try:
                progress = st.progress(0, text="Extracting factual claims...")
                results = run_verification(answer, chunks, progress)
                progress.empty()
                st.success(f"Verification complete · {len(results)} claim(s) analysed.")
                save_check([f.name for f in uploaded], results)
                st.session_state["last_results"] = results
                st.session_state["last_files"] = [f.name for f in uploaded]
            except json.JSONDecodeError:
                st.error("The AI returned an unreadable response. Please retry once.")
            except Exception as exc:
                st.error("Verification could not be completed.")
                st.code(str(exc))
    if st.session_state.get("last_results"):
        st.divider()
        show_results(st.session_state["last_results"])

elif page == "Evaluation lab":
    hero("Evaluation lab",
         "Measure the checker on examples with known labels. Use a labelled CSV to calculate real accuracy instead of claiming an untested score.")
    st.markdown("### Required CSV format")
    st.code("claim,expected_verdict\n'N-gram models use previous words to predict the next word',SUPPORTED\n'Laplace smoothing gives zero probability to unseen words',CONTRADICTED")
    st.caption("Expected labels must be SUPPORTED, CONTRADICTED or UNVERIFIABLE. Verify each row against the same uploaded notes.")
    eval_file = st.file_uploader("Upload labelled evaluation CSV", type=["csv"], key="eval_csv")
    eval_pdfs = st.file_uploader("Upload the reference PDFs for this test", type=["pdf"], accept_multiple_files=True, key="eval_pdfs")
    if eval_file:
        try:
            test_df = pd.read_csv(eval_file)
            st.dataframe(test_df.head(10), use_container_width=True, hide_index=True)
            required = {"claim", "expected_verdict"}
            if not required.issubset(test_df.columns):
                st.error("CSV must contain claim and expected_verdict columns.")
            elif eval_pdfs and st.button("Run labelled evaluation", type="primary"):
                with st.spinner("Building evidence index..."):
                    eval_chunks = extract_pdf_pages(eval_pdfs)
                predictions = []
                bar = st.progress(0)
                for i, row in test_df.iterrows():
                    ev = retrieve_evidence(str(row["claim"]), eval_chunks)
                    pred = verify_claim(str(row["claim"]), ev)
                    predictions.append(pred["verdict"])
                    bar.progress((i + 1) / len(test_df))
                expected = test_df["expected_verdict"].astype(str).str.upper().tolist()
                accuracy = sum(p == e for p, e in zip(predictions, expected)) / max(1, len(expected))
                st.metric("Measured accuracy", f"{accuracy*100:.1f}%", f"{sum(p == e for p,e in zip(predictions, expected))} correct of {len(expected)}")
                result_df = test_df.copy()
                result_df["predicted_verdict"] = predictions
                result_df["correct"] = [p == e for p, e in zip(predictions, expected)]
                st.dataframe(result_df, use_container_width=True, hide_index=True)
                st.download_button("Download evaluation results", result_df.to_csv(index=False), "noteproof_evaluation.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not read evaluation CSV: {exc}")
    else:
        st.info("Start with 30–50 manually labelled claims, including supported, contradicted and unverifiable cases. Report the measured result honestly.")

elif page == "History":
    hero("Verification history",
         "A local record of previous checks from this browser session and project folder.")
    history = load_history()
    if history.empty:
        st.info("No verification runs saved yet. Complete a check to see it here.")
    else:
        total = int(history["total"].sum())
        st.metric("Total claims checked", total)
        st.dataframe(history, use_container_width=True, hide_index=True)
        selected_id = st.selectbox("Open a saved verification run", history["id"].tolist())
        con = sqlite3.connect(DB_PATH)
        row = con.execute("SELECT result_json FROM checks WHERE id=?", (int(selected_id),)).fetchone()
        con.close()
        if row:
            saved_results = json.loads(row[0])
            show_results(saved_results)
        if st.button("Clear all local history"):
            con = sqlite3.connect(DB_PATH)
            con.execute("DELETE FROM checks")
            con.commit()
            con.close()
            st.rerun()

st.markdown("---")
st.markdown('<div class="small-muted">NOTEPROOF · AI-generated information deserves evidence, not just confidence. · Gemini API powered</div>', unsafe_allow_html=True)
