"""
app.py
──────
Streamlit web interface for the LLM-Powered Multi-Agent NOC System.
Deployed on HuggingFace Spaces (free tier, permanent URL).

Local/Colab run:
    streamlit run app.py

HuggingFace Spaces:
    Deployed automatically when pushed to the Space repository.
    Set HF_TOKEN as a Space Secret for live LLM mode.
    Without HF_TOKEN the app runs in DEMO mode (full pipeline,
    synthetic outputs — no API key required).

WHY Streamlit rather than Flask?
  Streamlit requires zero boilerplate HTML/JS — the entire UI is
  expressed in Python. Session state (st.session_state) persists
  results across reruns without a database. For a demonstration
  app where the primary goal is showing agent behaviour clearly,
  Streamlit is the right trade-off between speed and capability.

WHY HuggingFace Spaces for hosting?
  Spaces provides a free, permanent public URL with 2 vCPU and
  16 GB RAM — sufficient for sentence-transformers + FAISS + the
  Streamlit UI. Unlike Colab proxy URLs, the Spaces URL never
  changes, making it safe to include in academic submissions.
"""

import os
import sys
import time
import json
from datetime import datetime, timezone

import streamlit as st

# ── Path setup — works locally, in Colab, and on HuggingFace Spaces ───────
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

# ── FAISS cache: redirect to /tmp on Spaces (read-only filesystem) ─────────
# HuggingFace Spaces has a read-only app directory at runtime.
# /tmp is always writable and persists for the lifetime of the container.
# This env var is read by agents/retrieval.py to set INDEX_PATH.
if not os.environ.get("FAISS_INDEX_DIR"):
    os.environ["FAISS_INDEX_DIR"] = "/tmp/faiss_cache"
os.makedirs(os.environ["FAISS_INDEX_DIR"], exist_ok=True)

# ── Page config — must be first Streamlit call ─────────────────────────────
st.set_page_config(
    page_title="NOC Agent System",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)



# ── Imports (after path setup) ─────────────────────────────────────────────
try:
    from agents.message import AgentMessage
    from agents.orchestrator import OrchestratorAgent
    AGENTS_LOADED = True
except ImportError as e:
    AGENTS_LOADED = False
    IMPORT_ERROR = str(e)

# ══════════════════════════════════════════════════════════════════════════
# THEME & CUSTOM CSS
# Design language: dark navy operations room aesthetic — matching the
# presentation deck palette (navy #0D2B4E, teal #0A8A8A, amber #E8A020).
# Typography: system-ui for UI chrome, monospace for log output.
# One bold element: the live execution log with coloured phase tags.
# Everything else is quiet and functional.
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Base ──────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0b1f35;
    color: #dce8f0;
    font-family: 'Segoe UI', system-ui, sans-serif;
}

[data-testid="stSidebar"] {
    background-color: #0d2b4e;
    border-right: 1px solid #1a3a5c;
}

[data-testid="stSidebar"] * { color: #c8dce8 !important; }

/* ── Header strip ───────────────────────────────────────────────────── */
.noc-header {
    background: linear-gradient(90deg, #0a8a8a 0%, #0d2b4e 60%);
    border-radius: 10px;
    padding: 18px 28px 14px;
    margin-bottom: 24px;
}
.noc-header h1 {
    color: #ffffff;
    font-size: 1.55rem;
    font-weight: 700;
    margin: 0 0 4px;
    letter-spacing: -0.3px;
}
.noc-header p {
    color: #a8d4d4;
    font-size: 0.85rem;
    margin: 0;
}

/* ── Section labels ─────────────────────────────────────────────────── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    color: #0fb5b5;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* ── Input panel ────────────────────────────────────────────────────── */

/* Field labels (Severity, Segment / Location, etc.) */
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stSelectbox"] label,
.stTextInput label, .stTextArea label, .stSelectbox label {
    color: #7ec8d8 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}

/* Input and textarea — background, text, border — all variants */
.stTextInput > div > div > input,
.stTextInput input,
[data-testid="stTextInput"] input,
.stTextArea > div > div > textarea,
.stTextArea textarea,
[data-testid="stTextArea"] textarea {
    background-color: #0f2845 !important;
    border: 1px solid #1e4a6e !important;
    color: #dce8f0 !important;
    border-radius: 6px !important;
    font-size: 0.88rem !important;
    -webkit-text-fill-color: #dce8f0 !important;
}

/* Override Streamlit's own focus/active states */
.stTextInput > div,
.stTextArea > div,
[data-testid="stTextInput"] > div,
[data-testid="stTextArea"] > div {
    background-color: #0f2845 !important;
    border-radius: 6px !important;
}

/* Placeholder text — all selector variants to cover Streamlit's DOM */
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder,
.stTextInput input::placeholder,
.stTextArea textarea::placeholder,
input::placeholder,
textarea::placeholder,
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder {
    color: #7ec8d8 !important;
    opacity: 1 !important;
}

/* Typed text inside inputs and textareas */
.stTextInput input,
.stTextArea textarea,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    color: #dce8f0 !important;
    caret-color: #0fb5b5 !important;
}

/* Focus ring */
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #0a8a8a !important;
    box-shadow: 0 0 0 2px rgba(10,138,138,0.25) !important;
}

/* Selectbox — container, selected value, and dropdown */
.stSelectbox > div > div {
    background-color: #0f2845 !important;
    border: 1px solid #1e4a6e !important;
    border-radius: 6px !important;
}
.stSelectbox > div > div > div {
    color: #dce8f0 !important;
    font-size: 0.88rem !important;
}
/* Dropdown arrow icon */
.stSelectbox svg { fill: #7ec8d8 !important; }

/* ── Run button ─────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #0a8a8a, #077070) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 1.8rem !important;
    width: 100% !important;
    transition: opacity 0.15s !important;
    letter-spacing: 0.2px !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Execution log ──────────────────────────────────────────────────── */
.log-box {
    background: #060f1a;
    border: 1px solid #1a3a5c;
    border-radius: 8px;
    padding: 14px 16px;
    font-family: 'Cascadia Code', 'Fira Code', 'Courier New', monospace;
    font-size: 0.75rem;
    line-height: 1.7;
    max-height: 420px;
    overflow-y: auto;
    color: #8fbfcf;
}
.log-phase   { color: #0fb5b5; font-weight: 700; }
.log-msg     { color: #e8a020; }
.log-ok      { color: #27ae60; }
.log-warn    { color: #e8a020; }
.log-error   { color: #e74c3c; }

/* ── Report card ────────────────────────────────────────────────────── */
.report-card {
    background: #0d2b4e;
    border: 1px solid #1e4a6e;
    border-radius: 10px;
    padding: 22px 24px;
}
.report-card h3 {
    color: #0fb5b5;
    font-size: 0.95rem;
    font-weight: 700;
    margin: 0 0 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid #1e4a6e;
}

/* ── Metric badges ──────────────────────────────────────────────────── */
.badge-row { display: flex; gap: 10px; margin-bottom: 18px; flex-wrap: wrap; }
.badge {
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    line-height: 1.3;
    min-width: 80px;
    text-align: center;
}
.badge-label { font-size: 0.65rem; font-weight: 400; opacity: 0.8; display: block; margin-bottom: 2px; }
.badge-green  { background: #0d3320; border: 1px solid #27ae60; color: #2ecc71; }
.badge-amber  { background: #2d1f00; border: 1px solid #e8a020; color: #f0b429; }
.badge-red    { background: #2d0a0a; border: 1px solid #e74c3c; color: #ff6b6b; }
.badge-teal   { background: #002d2d; border: 1px solid #0a8a8a; color: #0fb5b5; }
.badge-blue   { background: #0a1a2d; border: 1px solid #2e6da4; color: #6ab0e8; }

/* ── Root cause block ───────────────────────────────────────────────── */
.root-cause {
    background: #0a1a2d;
    border-left: 3px solid #0a8a8a;
    border-radius: 0 6px 6px 0;
    padding: 12px 16px;
    margin-bottom: 16px;
    font-size: 0.88rem;
    color: #c8e0f0;
    line-height: 1.5;
}

/* ── Actions list ───────────────────────────────────────────────────── */
.action-item {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    padding: 7px 0;
    border-bottom: 1px solid #132a44;
    font-size: 0.85rem;
    color: #c0d8ea;
}
.action-item:last-child { border-bottom: none; }
.action-num {
    background: #0a8a8a;
    color: white;
    border-radius: 50%;
    width: 20px;
    height: 20px;
    min-width: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 700;
    margin-top: 1px;
}

/* ── KB match chips ─────────────────────────────────────────────────── */
.kb-chip {
    display: inline-block;
    background: #091e33;
    border: 1px solid #1e4a6e;
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 0.72rem;
    color: #6ab0e8;
    margin: 3px 3px 3px 0;
    font-family: monospace;
}

/* ── Message trace ──────────────────────────────────────────────────── */
.trace-item {
    display: flex;
    gap: 8px;
    align-items: flex-start;
    padding: 5px 0;
    border-bottom: 1px solid #0d1f33;
    font-size: 0.75rem;
    font-family: monospace;
    color: #7aa8c0;
}
.trace-num { color: #e8a020; min-width: 36px; font-weight: 700; }
.trace-arrow { color: #0fb5b5; }

/* ── Empty state ────────────────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 48px 24px;
    color: #3a6080;
}
.empty-state .icon { font-size: 2.5rem; margin-bottom: 12px; }
.empty-state p { font-size: 0.9rem; margin: 0; }

/* ── Preset buttons ─────────────────────────────────────────────────── */
.stButton.preset > button {
    background: #091e33 !important;
    border: 1px solid #1e4a6e !important;
    color: #6ab0e8 !important;
    font-size: 0.8rem !important;
    padding: 0.4rem 0.8rem !important;
    border-radius: 6px !important;
    width: 100% !important;
    text-align: left !important;
    font-weight: 400 !important;
}
.stButton.preset > button:hover {
    border-color: #0a8a8a !important;
    color: #0fb5b5 !important;
    opacity: 1 !important;
}

/* ── st.metric — label, value, delta ───────────────────────────────── */
[data-testid="stMetricLabel"] {
    color: #7ec8d8 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
}
[data-testid="metric-container"] {
    background-color: #0d2b4e !important;
    border: 1px solid #1e4a6e !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
}

/* ── Right panel text — body, lists, headings ───────────────────────── */
[data-testid="stVerticalBlock"] p,
[data-testid="stVerticalBlock"] li {
    color: #c8dce8 !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
}
[data-testid="stVerticalBlock"] h4 {
    color: #dce8f0 !important;
    font-weight: 700 !important;
}
[data-testid="stVerticalBlock"] strong {
    color: #ffffff !important;
}
[data-testid="stCaptionContainer"] p {
    color: #6a90a8 !important;
}

/* ── Message trace code block ───────────────────────────────────────── */
pre, [data-testid="stCode"] pre {
    background-color: #060f1a !important;
    border: 1px solid #1a3a5c !important;
    border-radius: 8px !important;
}
pre code, [data-testid="stCode"] code {
    color: #8fbfcf !important;
    font-size: 0.75rem !important;
    line-height: 1.7 !important;
}

/* ── Dividers ───────────────────────────────────────────────────────── */
hr { border-color: #1e4a6e !important; margin: 12px 0 !important; }

/* ── Alert boxes ────────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 8px !important; font-weight: 500 !important; }
[data-testid="stAlert"] p { color: inherit !important; }

/* ── Scrollbar ──────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0b1f35; }
::-webkit-scrollbar-thumb { background: #1e4a6e; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PRESET SCENARIOS
# ══════════════════════════════════════════════════════════════════════════

PRESETS = {
    "🔴  P1 MPLS Circuit Fault": {
        "goal": "Diagnose P1 MPLS circuit fault — Segment: DXB-AUH-01",
        "severity": "P1",
        "segment": "DXB-AUH-01",
        "pe_router": "DXB-PE-01",
        "symptoms": "BGP keepalive failure, MPLS label forwarding errors, packet loss >30%",
        "change_ref": "CHG-4421",
    },
    "🟠  BGP Route Leak": {
        "goal": "Investigate BGP route leak from customer AS 65001 to transit peers",
        "severity": "P1",
        "segment": "Core — AS 65001",
        "pe_router": "AUH-PE-02",
        "symptoms": "Customer prefixes visible in external BGP looking glass, AS path leak through AS 50001",
        "change_ref": "CHG-4389",
    },
    "⚫  Fibre Cut — E311 Corridor": {
        "goal": "Investigate fibre cut on E311 highway corridor — DXB-AUH optical span",
        "severity": "P1",
        "segment": "DXB-AUH Optical",
        "pe_router": "N/A",
        "symptoms": "Complete loss of signal on DXB-AUH span, OTDR shows 40 dB loss at 12.4 km",
        "change_ref": "N/A",
    },
    "🟡  OSPF Neighbour Flap": {
        "goal": "Investigate OSPF neighbour flap on core ring — DXB ring segment",
        "severity": "P2",
        "segment": "DXB Core Ring",
        "pe_router": "DXB-P-03",
        "symptoms": "OSPF adjacency flapping every 90s, BFD timer alerts, minor traffic impact",
        "change_ref": "CHG-4401",
    },
    "🔵  DWDM Channel Degradation": {
        "goal": "Investigate elevated BER on DWDM channels — DXB-SHJ optical link",
        "severity": "P2",
        "segment": "DXB-SHJ DWDM",
        "pe_router": "N/A",
        "symptoms": "BER >1e-9 on channels 32–36, EDFA gain tilt detected post-maintenance",
        "change_ref": "CHG-4415",
    },
}


# ══════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "result":        None,
        "log_lines":     [],
        "running":       False,
        "active_preset": None,
        "goal":          "",
        "severity":      "P1",
        "segment":       "",
        "pe_router":     "",
        "symptoms":      "",
        "change_ref":    "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════════════════════
# LOG CAPTURE — intercept agent print() calls in real time
# ══════════════════════════════════════════════════════════════════════════

class LogCapture:
    """
    Intercepts all print() calls during pipeline execution and stores
    them in st.session_state.log_lines so the UI can display them.

    WHY intercept print() rather than reading agent.log lists after the run?
      Streamlit reruns the script on every interaction. Capturing to
      session_state during the run means the log persists across reruns
      and is available for the final render without re-running the pipeline.
    """
    def __init__(self):
        self._original = sys.stdout
        self.lines = []

    def write(self, text):
        self._original.write(text)   # still prints to Colab cell
        stripped = text.strip()
        if stripped:
            self.lines.append(stripped)
            st.session_state.log_lines = list(self.lines)

    def flush(self):
        self._original.flush()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *_):
        sys.stdout = self._original


# ══════════════════════════════════════════════════════════════════════════
# LOG RENDERER — colour-coded by phase tag
# ══════════════════════════════════════════════════════════════════════════

def _classify_line(line: str) -> str:
    """Return a CSS class based on the content of a log line."""
    u = line.upper()
    if any(x in u for x in ["[PHASE", "===", "PIPELINE COMPLETE"]):
        return "log-phase"
    if "MSG #" in u:
        return "log-msg"
    if any(x in u for x in ["ERROR", "FAILED", "TIMEOUT"]):
        return "log-error"
    if any(x in u for x in ["WARNING", "WARN", "DEMO MODE"]):
        return "log-warn"
    if any(x in u for x in ["✓", "COMPLETE", "PASSED", "REPORT COMPLETE"]):
        return "log-ok"
    return ""


def render_log(lines: list) -> str:
    if not lines:
        return '<div class="log-box" style="color:#3a6080;font-style:italic;">Waiting for pipeline to run…</div>'
    html_lines = []
    for line in lines:
        cls = _classify_line(line)
        escaped = (line
                   .replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;"))
        if cls:
            html_lines.append(f'<span class="{cls}">{escaped}</span>')
        else:
            html_lines.append(escaped)
    body = "<br>".join(html_lines)
    return f'<div class="log-box">{body}</div>'


# ══════════════════════════════════════════════════════════════════════════
# REPORT RENDERER
# ══════════════════════════════════════════════════════════════════════════

def render_report(rd: dict, report_md: str, trace: list):
    """
    Render the resolution report using Streamlit native components.

    WHY native components rather than st.markdown(html)?
      Colab's proxy enforces a Content Security Policy that strips
      custom CSS classes from injected HTML, causing raw tags to appear
      instead of styled elements. Streamlit's own st.metric, st.info,
      st.success etc. are rendered server-side and are unaffected by
      the proxy's CSP. This is the correct approach for Colab hosting.
    """

    conf     = rd.get("confidence", 0)
    conf_pct = int(conf * 100)
    esc      = rd.get("escalation_required", False)
    sla      = rd.get("sla_risk", "MEDIUM")
    tier     = rd.get("escalation_tier", "N/A")
    sev      = rd.get("severity", "P2")
    inc_type = rd.get("incident_type", "UNKNOWN")
    root     = rd.get("root_cause", "N/A")
    actions  = rd.get("recommended_actions", [])
    kb_ids   = rd.get("kb_matches", [])

    st.markdown("#### 📋 Resolution Report")
    st.divider()

    # ── Row 1: Key metrics using st.metric ────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Severity",   sev)
    c2.metric("Confidence", f"{conf_pct}%")
    c3.metric("Escalate",   "YES ⚠️" if esc else "NO ✅")
    c4.metric("SLA Risk",   sla)
    c5.metric("Type",       inc_type)
    c6.metric("Tier",       tier.replace("Tier-", "T"))

    st.divider()

    # ── Root cause ────────────────────────────────────────────────────
    st.markdown("**🔍 Root Cause**")
    if sla == "HIGH" or esc:
        st.error(root, icon="🔴")
    elif sla == "MEDIUM":
        st.warning(root, icon="🟠")
    else:
        st.info(root, icon="🔵")

    # ── Recommended actions ───────────────────────────────────────────
    st.markdown("**⚡ Recommended Actions**")
    for i, action in enumerate(actions, 1):
        st.markdown(f"{i}. {action}")

    # ── Escalation callout ────────────────────────────────────────────
    if esc:
        st.error(f"**Escalation Required → {tier}**", icon="🚨")
    else:
        st.success("No escalation required — Tier-1 NOC can resolve", icon="✅")

    st.divider()

    # ── Knowledge base matches ────────────────────────────────────────
    st.markdown("**📚 Knowledge Base Matches**")
    if kb_ids:
        st.markdown("  ".join([f"`{k}`" for k in kb_ids]))
    else:
        st.caption("No KB matches retrieved")

    # ── Inter-agent message trace ─────────────────────────────────────
    st.markdown("**🔗 Inter-Agent Message Trace**")
    if trace:
        trace_text = "\n".join(trace)
        st.code(trace_text, language=None)
    else:
        st.caption("No trace available")

    st.divider()

    # ── Full report download ──────────────────────────────────────────
    with st.expander("📄  Full Markdown Report", expanded=False):
        st.markdown(report_md)
        st.download_button(
            label="⬇️  Download report (.md)",
            data=report_md,
            file_name=f"noc_report_{inc_type.lower()}_{datetime.now().strftime('%H%M%S')}.md",
            mime="text/markdown",
        )


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("### Quick Scenarios")
        st.markdown(
            "<p style='font-size:0.78rem;color:#6a90a8;margin-bottom:12px'>"
            "Load a pre-built incident into the form</p>",
            unsafe_allow_html=True
        )

        for label, preset in PRESETS.items():
            if st.button(label, key=f"preset_{label}"):
                st.session_state.active_preset = label
                st.session_state.goal       = preset["goal"]
                st.session_state.severity   = preset["severity"]
                st.session_state.segment    = preset["segment"]
                st.session_state.pe_router  = preset["pe_router"]
                st.session_state.symptoms   = preset["symptoms"]
                st.session_state.change_ref = preset["change_ref"]
                st.session_state.result     = None
                st.session_state.log_lines  = []
                st.rerun()

        st.markdown("---")
        st.markdown("### HuggingFace Token")
        st.markdown(
            "<p style='font-size:0.78rem;color:#6a90a8'>"
            "Optional — leave blank for demo mode</p>",
            unsafe_allow_html=True
        )
        hf_token = st.text_input(
            "HF Token",
            value=os.environ.get("HF_TOKEN", ""),
            type="password",
            label_visibility="collapsed",
            placeholder="hf_xxxxxxxxxxxx",
        )
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            st.success("Token set — live mode active", icon="✅")
        else:
            st.info("Running in demo mode", icon="ℹ️")

        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "<p style='font-size:0.78rem;color:#6a90a8'>"
            "MSc Artificial Intelligence<br>"
            "University of Essex Online<br><br>"
            "5-agent pipeline:<br>"
            "Orchestrator → Planner<br>"
            "→ Retrieval → Analyser<br>"
            "→ Reporter<br><br>"
            "<b style='color:#0fb5b5'>Hosted on HuggingFace Spaces</b><br>"
            "Free tier · Permanent URL<br>"
            "May take ~30s to wake up<br>"
            "after 48h of inactivity.</p>",
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════

def main():

    if not AGENTS_LOADED:
        st.error(f"Could not load agent modules: {IMPORT_ERROR}")
        st.info("Make sure PROJECT_ROOT is in sys.path before launching Streamlit.")
        return

    render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="noc-header">
        <h1>🔴 NOC Multi-Agent Incident Management</h1>
        <p>LLM-powered planning agent · BDI architecture · RAG knowledge retrieval · Real-time execution trace</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Two-column layout ─────────────────────────────────────────────────
    left, right = st.columns([1, 1.35], gap="large")

    # ── LEFT: Input form ──────────────────────────────────────────────────
    with left:
        st.markdown('<div class="section-label">Incident Details</div>', unsafe_allow_html=True)

        goal = st.text_area(
            "Incident Goal",
            value=st.session_state.goal,
            height=90,
            placeholder="Describe the incident goal, e.g. Diagnose P1 MPLS circuit fault — Segment: DXB-AUH-01",
            label_visibility="collapsed",
        )

        c1, c2 = st.columns(2)
        with c1:
            severity = st.selectbox(
                "Severity",
                ["P1", "P2", "P3"],
                index=["P1", "P2", "P3"].index(st.session_state.severity),
            )
        with c2:
            segment = st.text_input(
                "Segment / Location",
                value=st.session_state.segment,
                placeholder="e.g. DXB-AUH-01",
            )

        pe_router = st.text_input(
            "PE / Core Router",
            value=st.session_state.pe_router,
            placeholder="e.g. DXB-PE-01",
        )

        symptoms = st.text_area(
            "Symptoms",
            value=st.session_state.symptoms,
            height=100,
            placeholder="Describe observed symptoms, alarms, or error conditions…",
        )

        change_ref = st.text_input(
            "Change Reference (optional)",
            value=st.session_state.change_ref,
            placeholder="e.g. CHG-4421",
        )

        # ── Run button ─────────────────────────────────────────────────
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        run_clicked = st.button("▶  Run Agent Pipeline", use_container_width=True)

        # ── Execution log ───────────────────────────────────────────────
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Execution Log</div>', unsafe_allow_html=True)
        log_placeholder = st.empty()
        log_placeholder.markdown(
            render_log(st.session_state.log_lines),
            unsafe_allow_html=True
        )

    # ── RIGHT: Results ────────────────────────────────────────────────────
    with right:
        st.markdown('<div class="section-label">Analysis & Resolution</div>', unsafe_allow_html=True)
        result_placeholder = st.empty()

        if st.session_state.result is None:
            result_placeholder.markdown("""
            <div class="empty-state">
                <div class="icon">📡</div>
                <p>Submit an incident on the left to run the agent pipeline.<br>
                Results and the inter-agent message trace will appear here.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            with result_placeholder.container():
                render_report(
                    st.session_state.result["report_dict"],
                    st.session_state.result["report_md"],
                    st.session_state.result["trace"],
                )

    # ══════════════════════════════════════════════════════════════════════
    # PIPELINE EXECUTION (triggered by Run button)
    # ══════════════════════════════════════════════════════════════════════

    if run_clicked:
        if not goal.strip():
            st.warning("Please enter an incident goal before running.", icon="⚠️")
            st.stop()

        # Build incident dict from form fields
        incident = {"severity": severity}
        if segment:    incident["segment"]    = segment
        if pe_router:  incident["pe_router"]  = pe_router
        if symptoms:   incident["symptoms"]   = symptoms
        if change_ref and change_ref != "N/A":
            incident["change_reference"] = change_ref

        # Clear previous results
        st.session_state.log_lines = []
        st.session_state.result    = None

        hf_token = os.environ.get("HF_TOKEN", "") or None

        with st.spinner("Agent pipeline running…"):
            try:
                orch = OrchestratorAgent(hf_token=hf_token)

                with LogCapture() as capture:
                    msg = AgentMessage(
                        sender="system",
                        recipient="orchestrator",
                        payload={"goal": goal.strip(), "incident": incident},
                    )
                    response = orch.run(msg)

                # Store results in session state
                st.session_state.result = {
                    "report_dict": response.payload["report_dict"],
                    "report_md":   response.payload["report_md"],
                    "trace":       response.payload.get("message_trace", []),
                }
                st.session_state.log_lines = capture.lines

            except Exception as e:
                st.error(f"Pipeline error: {e}", icon="🔴")
                st.session_state.log_lines.append(f"[ERROR] {e}")

        # Update log display and results
        log_placeholder.markdown(
            render_log(st.session_state.log_lines),
            unsafe_allow_html=True
        )
        with result_placeholder.container():
            if st.session_state.result:
                render_report(
                    st.session_state.result["report_dict"],
                    st.session_state.result["report_md"],
                    st.session_state.result["trace"],
                )


if __name__ == "__main__":
    main()
