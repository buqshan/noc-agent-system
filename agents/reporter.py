"""
agents/reporter.py
──────────────────
The ReporterAgent formats the combined outputs from all upstream agents
into a structured Markdown incident resolution report.

WHY separate reporting from analysis?
  The Reporter receives a completed analysis dict and has no knowledge
  of HOW the analysis was produced. This separation means:
    - Output format (Markdown, JSON, PDF) can change without touching
      the reasoning logic in AnalyserAgent.
    - The report structure is independently unit-testable with mock data.
    - Future versions could support multiple output formats via a
      Strategy pattern (one Reporter per format).

WHY Markdown as the primary output?
  Markdown satisfies two requirements simultaneously:
    (a) Human-readable in any editor or terminal — a NOC engineer
        can action the report without any rendering step.
    (b) Machine-parseable — the structured sections are recoverable
        by simple string operations, enabling downstream automation.
"""

from datetime import datetime, timezone
from agents.base import BaseAgent
from agents.message import AgentMessage


# SLA definitions by severity (minutes to resolution target)
# Drawn from ITIL v4 incident severity classifications
SLA_TARGETS = {
    "P1": 240,    # 4 hours
    "P2": 480,    # 8 hours
    "P3": 1440,   # 24 hours
}


def _severity_from_incident(incident: dict) -> str:
    """Extract priority/severity from the incident dict, defaulting to P2."""
    sev = incident.get("severity", incident.get("priority", "P2")).upper()
    if sev not in ("P1", "P2", "P3"):
        sev = "P2"
    return sev


class ReporterAgent(BaseAgent):
    """
    Produces a structured Markdown NOC incident resolution report.

    Input payload keys:  analysis (dict), incident_type (str), goal (str),
                         incident (dict), plan (list), retrieved_docs (list)
    Output payload keys: report_md (str), report_dict (dict)
    """

    def __init__(self):
        super().__init__("reporter")

    def run(self, message: AgentMessage) -> AgentMessage:
        analysis      = message.payload.get("analysis", {})
        inc_type      = message.payload.get("incident_type", "UNKNOWN")
        goal          = message.payload.get("goal", "N/A")
        incident      = message.payload.get("incident", {})
        plan          = message.payload.get("plan", [])
        retrieved     = message.payload.get("retrieved_docs", [])

        self._log("Composing incident resolution report...")

        severity = _severity_from_incident(incident)
        sla_mins = SLA_TARGETS.get(severity, 480)
        now_utc  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # ── Build the Markdown report ─────────────────────────────────────
        lines = []

        lines.append("# NOC INCIDENT RESOLUTION REPORT")
        lines.append(f"**Generated:** {now_utc}")
        lines.append(f"**Severity:** {severity}  |  **SLA Target:** {sla_mins} minutes")
        lines.append(f"**Incident Type:** {inc_type}")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## 1. Incident Goal")
        lines.append(f"> {goal}")
        lines.append("")

        if incident:
            lines.append("## 2. Incident Details")
            for k, v in incident.items():
                lines.append(f"- **{k.replace('_', ' ').title()}:** {v}")
            lines.append("")

        lines.append("## 3. Diagnostic Plan Executed")
        for step in plan:
            lines.append(f"- **Step {step['step']}:** {step['action']}")
        lines.append("")

        lines.append("## 4. Knowledge Base Matches")
        for doc in retrieved:
            sim_pct = int(doc.get("similarity", 0) * 100)
            lines.append(
                f"- [{doc['id']}] **{doc['title']}** "
                f"(relevance: {sim_pct}%)"
            )
        lines.append("")

        lines.append("## 5. Root Cause Analysis")
        conf_pct = int(analysis.get("confidence", 0) * 100)
        conf_label = (
            "HIGH" if conf_pct >= 80 else
            "MEDIUM" if conf_pct >= 50 else
            "LOW"
        )
        lines.append(f"**Root Cause:** {analysis.get('root_cause', 'N/A')}")
        lines.append("")
        lines.append(
            f"**Confidence:** {conf_pct}% ({conf_label})"
        )
        lines.append("")
        lines.append(f"**Technical Detail:**")
        lines.append(f"> {analysis.get('technical_detail', 'N/A')}")
        lines.append("")

        if analysis.get("fallback"):
            lines.append(
                f"> ⚠️ **Note:** LLM analysis unavailable "
                f"({analysis.get('fallback_reason', 'unknown reason')}). "
                f"Report is partial — manual analysis required."
            )
            lines.append("")

        lines.append("## 6. Recommended Actions")
        for i, action in enumerate(analysis.get("recommended_actions", []), 1):
            lines.append(f"{i}. {action}")
        lines.append("")

        lines.append("## 7. Escalation & SLA")
        esc = analysis.get("escalation_required", False)
        tier = analysis.get("escalation_tier", "N/A")
        risk = analysis.get("sla_risk", "MEDIUM")
        lines.append(f"- **Escalation Required:** {'✅ YES' if esc else '❌ NO'}")
        lines.append(f"- **Escalation Tier:** {tier}")
        lines.append(f"- **SLA Risk:** {risk}")
        lines.append("")

        lines.append("---")
        lines.append("*Report generated by LLM-Powered Multi-Agent NOC System*")
        lines.append("*MSc Artificial Intelligence — University of Essex Online*")

        report_md = "\n".join(lines)

        # ── Structured dict version for programmatic use ──────────────────
        report_dict = {
            "generated_at": now_utc,
            "severity": severity,
            "incident_type": inc_type,
            "root_cause": analysis.get("root_cause"),
            "confidence": analysis.get("confidence"),
            "escalation_required": analysis.get("escalation_required"),
            "escalation_tier": analysis.get("escalation_tier"),
            "sla_risk": analysis.get("sla_risk"),
            "recommended_actions": analysis.get("recommended_actions", []),
            "kb_matches": [d["id"] for d in retrieved],
            "plan_steps": len(plan),
        }

        self._log("Report complete.")
        self._log(f"  Root cause: {report_dict['root_cause'][:60]}...")
        self._log(f"  Escalation: {report_dict['escalation_required']}")
        self._log(f"  SLA risk:   {report_dict['sla_risk']}")

        return AgentMessage(
            sender=self.name,
            recipient="orchestrator",
            payload={"report_md": report_md, "report_dict": report_dict},
        )
