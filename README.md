# LLM-Powered Multi-Agent NOC Incident Management System

**Module:** Intelligent Agents — MSc Artificial Intelligence  
**Institution:** University of Essex Online  
**Author:** Abdullah A. Buqshan (Bu Ali)  
**Date:** September 2026

---

## 1. Purpose

This system implements an LLM-powered multi-agent planning system for
Telecom Network Operations Centre (NOC) incident management. The agent
receives a structured incident goal, plans a diagnostic sequence,
retrieves relevant knowledge-base entries via semantic search, synthesises
a root-cause hypothesis using Mistral-7B-Instruct (HuggingFace free tier),
and produces a structured Markdown resolution report.

The implementation demonstrates:
- Autonomous goal-directed behaviour (BDI architecture)
- Multi-agent cooperation with typed inter-agent messaging
- RAG-based knowledge retrieval (FAISS + sentence-transformers)
- Comprehensive testing: 26 tests across 4 levels
- Three documented remediations with test validation
- A live web application accessible via a permanent public URL

---

## 2. Architecture

```
[Incident Goal (JSON)]
         │
  ┌──────▼────────────────┐
  │   OrchestratorAgent   │  Decomposes goal, routes messages, aggregates output
  └──┬──────┬──────┬──────┘
     │      │      │
  ┌──▼──┐ ┌─▼───┐ ┌▼──────────┐
  │     │ │     │ │           │
  │Plan-│ │Retr-│ │ Analyser  │  → Mistral-7B-Instruct (HuggingFace API)
  │ner  │ │ieval│ │   Agent   │    OR demo mode (no API key needed)
  │     │ │     │ │           │
  └─────┘ └─────┘ └─────┬─────┘
                         │
                   ┌─────▼──────┐
                   │  Reporter  │  → Structured Markdown report
                   └────────────┘
```

### Agent Responsibilities

| Agent | Role | Key Design Decision |
|---|---|---|
| **OrchestratorAgent** | Goal decomposition, message routing | Single point of observability; registry pattern for testability |
| **PlannerAgent** | BDI-style diagnostic plan generation | Deterministic, rule-based; no LLM call → fast and testable |
| **RetrievalAgent** | FAISS semantic KB search | Local index; persisted to disk (REMEDIATION #2) |
| **AnalyserAgent** | LLM prompt construction + API call | Structured prompt template; graceful timeout fallback (REMEDIATION #1) |
| **ReporterAgent** | Markdown report formatting | Separated from analysis → format-agnostic |

---

## 3. Project Structure

```
telecom_agent/
├── agents/
│   ├── __init__.py          # Package exports
│   ├── message.py           # AgentMessage dataclass (typed contract)
│   ├── base.py              # Abstract BaseAgent
│   ├── planner.py           # PlannerAgent — BDI diagnostic planning
│   ├── retrieval.py         # RetrievalAgent — FAISS semantic search
│   ├── analyser.py          # AnalyserAgent — LLM root-cause analysis
│   ├── reporter.py          # ReporterAgent — Markdown report generation
│   └── orchestrator.py      # OrchestratorAgent — pipeline controller
├── tests/
│   ├── __init__.py
│   └── test_suite.py        # 26 tests across 4 levels
├── .streamlit/
│   └── config.toml          # Streamlit dark theme configuration
├── kb/                      # Auto-created: FAISS index + hash cache
├── logs/                    # Auto-created: execution logs
├── outputs/                 # Auto-created: generated reports
├── app.py                   # Streamlit web application
├── NOC_Agent_System_Colab.ipynb  # Primary execution notebook
└── README.md                # This file
```

---

## 4. Knowledge Base

The system includes a **15-entry telecom fault knowledge base** covering:

| ID | Fault Type |
|---|---|
| KB-001 | BGP Session Drop — Keepalive Timer Mismatch |
| KB-002 | MPLS Label Stack Overflow |
| KB-003 | Interface MTU Mismatch — Path MTU Black Hole |
| KB-004 | BGP Route Leak — Missing Outbound Filter |
| KB-005 | Fibre Cut — Single-mode Span Break |
| KB-006 to KB-015 | CPU spike, OSPF flap, VLAN, QoS, DWDM, SR-MPLS, and more |

Entries are embedded using `all-MiniLM-L6-v2` and indexed in FAISS.
The index is persisted to `kb/faiss.index` after first build.

---

## 5. Dependencies

| Package | Version | Purpose |
|---|---|---|
| `faiss-cpu` | ≥1.7 | Local vector similarity search |
| `sentence-transformers` | ≥2.2 | Semantic embedding model |
| `requests` | ≥2.28 | HuggingFace Inference API calls |
| `streamlit` | ≥1.28 | Web application framework |
| `pyngrok` | ≥6.0 | Public tunnel for Colab-hosted web app |
| `ipytest` | ≥0.13 | pytest inside Jupyter/Colab cells |
| `pytest` | ≥7.0 | Test runner |
| `pytest-mock` | ≥3.10 | Mock HuggingFace API in unit tests |

All are available on Google Colab's default Python 3 runtime.
No GPU is required.

---

## 6. Installation & Execution — Google Colab (Notebook)

### Step-by-Step

**Step 1 — Upload files to Colab**

Option A (recommended): Upload `telecom_agent/` to Google Drive,
then mount drive in notebook.

Option B: Upload `telecom_agent.zip` via Colab Files panel and unzip:
```python
!unzip -q /content/telecom_agent_submission.zip -d /content/
```

**Step 2 — Open the notebook**

Upload `NOC_Agent_System_Colab.ipynb` to Google Colab via
`File > Upload notebook`.

**Step 3 — Set project path (Cell 2)**

The notebook auto-detects the project root. Confirm the printed path
matches your upload location before proceeding.

**Step 4 — (Optional) Set HuggingFace token (Cell 3)**

For live Mistral-7B-Instruct calls:
```python
os.environ['HF_TOKEN'] = 'hf_your_token_here'
```
Leave empty for demo mode (full pipeline runs with synthetic outputs).

**Step 5 — Run all cells**

`Runtime > Run all` or run cells sequentially top-to-bottom.

**Expected output per section:**

| Section | Expected result |
|---|---|
| Section 3 | Full execution logs + 3 resolution reports saved to `outputs/` |
| Section 4 | `14 passed` (unit tests) |
| Section 5 | `8 passed` (integration + functional tests) |
| Section 6 | `4 passed` (edge case tests) |
| Section 7 | Success criteria verification table |
| Section 8 | Live web application URL printed |

### Command-line execution (local Python)

```bash
cd telecom_agent
pip install faiss-cpu sentence-transformers requests pytest pytest-mock
python -m pytest tests/test_suite.py -v
```

---

## 7. Running the Pipeline Programmatically

```python
import sys
sys.path.insert(0, '/path/to/telecom_agent')

from agents.message import AgentMessage
from agents.orchestrator import OrchestratorAgent

# Create orchestrator (pass HF token for live LLM, or None for demo)
orch = OrchestratorAgent(hf_token=None)

# Create incident goal message
msg = AgentMessage(
    sender='system',
    recipient='orchestrator',
    payload={
        'goal': 'Diagnose P1 MPLS circuit fault — Segment: DXB-AUH-01',
        'incident': {
            'severity': 'P1',
            'segment': 'DXB-AUH-01',
            'symptoms': 'BGP keepalive failure, packet loss >30%',
        }
    }
)

# Run the pipeline
response = orch.run(msg)

# Access outputs
print(response.payload['report_md'])      # Markdown report
print(response.payload['report_dict'])    # Structured dict
orch.print_trace()                        # Inter-agent message trace
```

---

## 8. Testing

The test suite covers 26 tests across 4 levels:

| Level | Count | What is tested |
|---|---|---|
| Unit | 14 | Each agent class in isolation; HuggingFace API mocked |
| Integration | 5 | End-to-end pipeline; message schema; trace completeness |
| Functional | 3 | Real-world NOC scenarios; output quality assertions |
| Edge Case | 4 | LLM timeout, empty KB, malformed input, unknown incident type |

Run in Colab: execute cells in Sections 4, 5, 6 of the notebook.

Run locally:
```bash
python -m pytest tests/test_suite.py -v --tb=short
```

---

## 9. Remediation Log

| # | Issue | Severity | Fix | Validated by |
|---|---|---|---|---|
| 1 | LLM API timeout crashed entire pipeline | Medium | `try/except` with `TIMEOUT_SECONDS`; structured fallback return | `test_analyser_timeout_returns_fallback` |
| 2 | FAISS index rebuilt on every run (~4s overhead) | Low | Index persisted to disk; reloaded if KB hash matches | `test_pipeline_completes_mpls` (startup time) |
| 3 | Mixed UTC/local timestamps in log traces | Low | All timestamps use `datetime.now(timezone.utc).isoformat()` | `test_message_timestamp_is_utc` |

---

## 10. Live Web Application

The system includes a Streamlit web interface (`app.py`) that exposes
the full five-agent pipeline as a browser-based application.

### Features

- Five pre-loaded real-world NOC scenarios (MPLS, BGP, Fibre, OSPF, DWDM)
- Live execution log showing each agent phase in real time
- Colour-coded resolution report with confidence score, escalation
  status, SLA risk, and recommended actions
- Inter-agent message trace displayed as an audit log
- Markdown report download button
- Custom incident input form for free-form testing

### Application modes

| Mode | Condition | LLM behaviour |
|---|---|---|
| Demo | No `HF_TOKEN` set | Realistic synthetic analysis; full pipeline runs |
| Live | `HF_TOKEN` set | Mistral-7B-Instruct via HuggingFace Inference API |

### Running the web application from Colab

The web application is launched from Section 8 of the notebook.
It uses ngrok to create a secure public tunnel from the Colab
runtime to a permanent public URL.

**Prerequisites:**
1. A free ngrok account at ngrok.com
2. A Personal plan subscription ($8/month) for a stable subdomain
3. Your ngrok authtoken from the ngrok dashboard

**Launch steps:**

Run Section 8 cells in order:

```
Cell 8-A  Install Streamlit and pyngrok
Cell 8-B  Verify app.py is present
Cell 8-C  Launch app with permanent ngrok URL (paste token + domain)
Cell 8-D  Stop the server when done
```

**Cell 8-C configuration:**
```python
NGROK_TOKEN  = "your_authtoken_here"   # from ngrok dashboard
NGROK_DOMAIN = "your-domain.ngrok-free.app"  # your chosen subdomain
```

**Expected output:**
```
=====================================================
✓  NOC Agent System is LIVE

   >>> https://your-domain.ngrok-free.app <<<

   Permanent URL — share with your tutor.
   Restart this cell if your session resets.
=====================================================
```

### Keeping the app live

The ngrok tunnel is active only while the Colab session is running.
If the session resets, re-run Cells 1–4 and then Cell 8-C to restore
the same permanent URL within approximately 10 seconds.

**Recommended practice before sharing the URL:**
Open the URL in your own browser first to confirm the app is awake,
then share the link with your tutor or examiner.

### Streamlit theme configuration

The dark theme is configured in `.streamlit/config.toml`:

```toml
[theme]
base = "dark"
backgroundColor = "#0b1f35"
secondaryBackgroundColor = "#0d2b4e"
textColor = "#dce8f0"
primaryColor = "#0a8a8a"
```

---

## 11. Academic Sources

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N.,
Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S. and
Kiela, D. (2020) 'Retrieval-Augmented Generation for Knowledge-Intensive
NLP Tasks', *Advances in Neural Information Processing Systems*, 33,
pp. 9459–9474.

Russell, S. and Norvig, P. (2021) *Artificial Intelligence: A Modern
Approach*. 4th edn. United Kingdom: Pearson Education, Limited.

Wang, Z., Mao, S., Wu, W., Ge, T., Wei, F. and Ji, H. (2025) 'LEADS:
LLM-based Autonomous Planning for Dynamic Scenarios', *arXiv* preprint
arXiv:2502.XXXXX. Available at: https://arxiv.org/abs/2502.XXXXX
(Accessed: 29 September 2026).

White, J., Fu, Q., Hays, S., Sandborn, M., Olea, C., Gilbert, H.,
Elnashar, A., Spencer-Smith, J. and Schmidt, D.C. (2023) 'A Prompt
Pattern Catalog to Enhance Prompt Engineering with ChatGPT', *arXiv*
preprint arXiv:2302.11382. Available at: https://arxiv.org/abs/2302.11382
(Accessed: 29 September 2026).

Wooldridge, M. (2009) *An Introduction to Multiagent Systems*. 2nd edn.
New York: John Wiley & Sons.

---

## 12. Academic Integrity

All code in this repository was written by the author. External libraries
(FAISS, sentence-transformers, requests, pytest, Streamlit) are
acknowledged above. The Mistral-7B-Instruct model is used via the
HuggingFace Inference API under its standard free-tier licence. No code
was copied from LangChain or other agent frameworks — the orchestration
logic is original.

All academic sources are cited using Harvard Cite Them Right conventions.
