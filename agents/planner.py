"""
agents/planner.py
─────────────────
The PlannerAgent applies a BDI (Belief-Desire-Intention) reasoning
pattern to convert a raw incident goal into a structured, ordered
diagnostic action plan.

WHY BDI framing?
  The BDI model (Bratman, 1987; Wooldridge, 2009, Ch.2) maps cleanly
  onto the NOC incident management process:
    • Beliefs  = current incident state (severity, segment, symptoms)
    • Desires  = outcomes sought (root cause identified, SLA preserved)
    • Intentions = the concrete diagnostic steps to execute now

WHY produce JSON output?
  The plan is serialised as a list of dicts so that:
    (a) the orchestrator can iterate it deterministically,
    (b) it is directly unit-testable without parsing free text,
    (c) it can be logged verbatim for the execution trace.

WHY NOT call the LLM here?
  Planning is separated from LLM inference deliberately.
  The plan is rule-based and deterministic, which means:
    - it is fast (no API call),
    - it is testable with fixed expected outputs,
    - plan validation can occur before any API spend is incurred.
  This follows the principle of 'fail early, fail cheaply'.
"""

from agents.base import BaseAgent
from agents.message import AgentMessage


# Diagnostic step templates keyed by incident type.
# WHY a dict of templates rather than if/elif chains?
#   Adding a new incident type requires only a new dict entry,
#   not touching the control flow. Open/Closed Principle.
PLAN_TEMPLATES = {
    "MPLS": [
        {"step": 1, "action": "Verify MPLS label forwarding table on PE routers"},
        {"step": 2, "action": "Check BGP session state between PE-CE peers"},
        {"step": 3, "action": "Inspect interface MTU settings on affected segment"},
        {"step": 4, "action": "Review recent change log for affected segment"},
    ],
    "BGP": [
        {"step": 1, "action": "Identify route leak scope — prefixes and AS path"},
        {"step": 2, "action": "Check route policy and filter configurations"},
        {"step": 3, "action": "Verify BGP community tags on affected prefixes"},
        {"step": 4, "action": "Assess downstream customer impact"},
    ],
    "FIBRE": [
        {"step": 1, "action": "Confirm physical layer alarm on affected span"},
        {"step": 2, "action": "Dispatch field team for visual inspection"},
        {"step": 3, "action": "Activate protection switching if available"},
        {"step": 4, "action": "Engage third-party carrier for joint troubleshooting"},
    ],
    "DEFAULT": [
        {"step": 1, "action": "Collect alarm and event log from affected NE"},
        {"step": 2, "action": "Isolate fault domain (access / core / transport)"},
        {"step": 3, "action": "Check recent maintenance or change activity"},
        {"step": 4, "action": "Escalate to Tier-2 NOC if unresolved in 30 min"},
    ],
}


def _detect_incident_type(goal: str) -> str:
    """
    Classify an incident goal string into one of the known types.

    WHY simple keyword matching rather than an NLP classifier?
      In a production NOC, incidents arrive via structured ITSM tickets
      with a defined 'fault_type' field. The keyword fallback covers
      free-text goals in the assessment demo without adding an
      unnecessary ML dependency to the planning step.
    """
    goal_upper = goal.upper()
    if "MPLS" in goal_upper:
        return "MPLS"
    elif "BGP" in goal_upper or "ROUTE" in goal_upper:
        return "BGP"
    elif "FIBRE" in goal_upper or "FIBER" in goal_upper or "OPTICAL" in goal_upper:
        return "FIBRE"
    return "DEFAULT"


class PlannerAgent(BaseAgent):
    """
    Converts a high-level incident goal into a stepwise diagnostic plan.

    Input payload keys:  goal (str), incident (dict)
    Output payload keys: plan (list[dict]), incident_type (str)
    """

    def __init__(self):
        super().__init__("planner")

    def run(self, message: AgentMessage) -> AgentMessage:
        self._log(f"Received goal: {message.payload.get('goal', 'N/A')}")

        goal = message.payload.get("goal", "")
        incident = message.payload.get("incident", {})

        incident_type = _detect_incident_type(goal)
        self._log(f"Classified incident type: {incident_type}")

        plan = PLAN_TEMPLATES[incident_type]
        self._log(f"Plan generated: {len(plan)} steps")

        for step in plan:
            self._log(f"  Step {step['step']}: {step['action']}")

        return AgentMessage(
            sender=self.name,
            recipient="orchestrator",
            payload={
                "plan": plan,
                "incident_type": incident_type,
                "goal": goal,
                "incident": incident,
            },
        )
