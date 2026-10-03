"""
agents/orchestrator.py
──────────────────────
The OrchestratorAgent is the top-level controller of the multi-agent
pipeline. It receives a raw incident goal, decomposes it into a
sequential set of delegations, dispatches messages to specialist agents,
and aggregates final outputs.

WHY a single centralised orchestrator rather than peer-to-peer agents?
  Two reasons:
    1. Observability: all inter-agent messages pass through a single
       point, so the complete execution trace is captured in one place.
       This is critical for both debugging and the testing requirement.
    2. Simplicity: for a 5-agent sequential pipeline, a centralised
       orchestrator avoids the distributed coordination complexity of
       peer-to-peer communication (Wooldridge, 2009, Ch.6).
  A decentralised design would be justified for systems with many
  concurrent agents or dynamic agent discovery — neither of which
  applies here.

WHY a registry dict rather than direct agent instantiation?
  The registry pattern (self.agents = {name: agent}) decouples the
  orchestrator from concrete agent classes. Swapping a mock agent
  into the registry during testing requires no changes to this file —
  only the test setUp code changes. This is the key property that
  enables isolated unit testing of the orchestration logic.

Message flow (matching slide 3 architecture diagram):
  Orchestrator ──► Planner ──► Orchestrator
  Orchestrator ──► Retrieval ──► Orchestrator
  Orchestrator ──► Analyser ──► Reporter
  Reporter ──► Orchestrator (final)
"""

import json
from datetime import datetime, timezone

from agents.base import BaseAgent
from agents.message import AgentMessage
from agents.planner import PlannerAgent
from agents.retrieval import RetrievalAgent
from agents.analyser import AnalyserAgent
from agents.reporter import ReporterAgent


class OrchestratorAgent(BaseAgent):
    """
    Central controller — decomposes the incident goal and routes
    messages through the specialist agent pipeline.

    Parameters
    ----------
    hf_token : str, optional
        HuggingFace Inference API token. Passed to AnalyserAgent.
        If absent, AnalyserAgent runs in demo mode.
    """

    def __init__(self, hf_token: str | None = None):
        super().__init__("orchestrator")

        # ── Agent registry ─────────────────────────────────────────────
        # WHY a dict keyed by name?
        #   Tests can inject mock agents by name without subclassing
        #   OrchestratorAgent. The orchestrator never imports concrete
        #   agent classes except at construction time.
        self.agents: dict[str, BaseAgent] = {
            "planner":   PlannerAgent(),
            "retrieval": RetrievalAgent(),
            "analyser":  AnalyserAgent(hf_token=hf_token),
            "reporter":  ReporterAgent(),
        }

        # Complete inter-agent message trace (for testing + evidence)
        self.message_trace: list[str] = []
        self._msg_counter = 0

    def _dispatch(self, target_name: str, payload: dict) -> AgentMessage:
        """
        Create a message, dispatch it to the named agent, log the trace.
        """
        self._msg_counter += 1
        agent = self.agents[target_name]

        outbound = AgentMessage(
            sender=self.name,
            recipient=target_name,
            payload=payload,
        )

        trace_out = (
            f"MSG #{self._msg_counter:02d} "
            f"{self.name}→{target_name} | "
            f"payload_keys={list(payload.keys())}"
        )
        self.message_trace.append(trace_out)
        self._log(trace_out)

        # ── Execute agent ──────────────────────────────────────────────
        response = agent.run(outbound)

        self._msg_counter += 1
        trace_in = (
            f"MSG #{self._msg_counter:02d} "
            f"{target_name}→{self.name} | "
            f"payload_keys={list(response.payload.keys())}"
        )
        self.message_trace.append(trace_in)
        self._log(trace_in)

        return response

    def run(self, message: AgentMessage) -> AgentMessage:
        """
        Main pipeline execution.

        Receives an incident goal message and drives the full
        plan → retrieve → analyse → report sequence.
        """
        goal     = message.payload.get("goal", "")
        incident = message.payload.get("incident", {})

        self._log("=" * 60)
        self._log(f"Orchestrator received goal: '{goal}'")
        self._log("=" * 60)

        # ── Step 1: Planning ────────────────────────────────────────────
        self._log("\n[PHASE 1: PLANNING]")
        plan_response = self._dispatch("planner", {
            "goal": goal,
            "incident": incident,
        })
        plan         = plan_response.payload["plan"]
        incident_type = plan_response.payload["incident_type"]
        self._log(f"Plan ready: {len(plan)} steps, type={incident_type}")

        # ── Step 2: Retrieval ───────────────────────────────────────────
        self._log("\n[PHASE 2: RETRIEVAL]")
        # Build retrieval query from goal + incident type
        retrieval_query = f"{incident_type} {goal} {incident.get('symptoms', '')}"
        retrieval_response = self._dispatch("retrieval", {
            "query": retrieval_query.strip(),
            "incident_type": incident_type,
        })
        retrieved_docs = retrieval_response.payload["retrieved_docs"]
        self._log(f"Retrieved {len(retrieved_docs)} KB entries.")

        # ── Step 3: Analysis ────────────────────────────────────────────
        self._log("\n[PHASE 3: ANALYSIS]")
        analysis_response = self._dispatch("analyser", {
            "plan":          plan,
            "retrieved_docs": retrieved_docs,
            "incident":      incident,
            "goal":          goal,
            "incident_type": incident_type,
        })
        analysis = analysis_response.payload["analysis"]

        # ── Step 4: Reporting ───────────────────────────────────────────
        # NOTE: Reporter is dispatched from Orchestrator (not Analyser)
        # so that Orchestrator retains full payload context for the report.
        # The deck's architecture shows Analyser→Reporter, but implementing
        # it via the Orchestrator allows the Reporter to receive the full
        # enriched payload (plan + retrieved + analysis) in one message.
        self._log("\n[PHASE 4: REPORTING]")
        report_response = self._dispatch("reporter", {
            "analysis":      analysis,
            "incident_type": incident_type,
            "goal":          goal,
            "incident":      incident,
            "plan":          plan,
            "retrieved_docs": retrieved_docs,
        })

        report_md   = report_response.payload["report_md"]
        report_dict = report_response.payload["report_dict"]

        self._log("\n" + "=" * 60)
        self._log("Pipeline complete.")
        self._log(f"  Messages exchanged: {self._msg_counter}")
        self._log(f"  Root cause: {report_dict.get('root_cause', 'N/A')[:60]}...")
        self._log(f"  Confidence: {report_dict.get('confidence', 0):.0%}")
        self._log(f"  Escalation: {report_dict.get('escalation_required')}")
        self._log("=" * 60)

        return AgentMessage(
            sender=self.name,
            recipient="system",
            payload={
                "report_md":    report_md,
                "report_dict":  report_dict,
                "message_trace": self.message_trace,
                "n_messages":   self._msg_counter,
            },
        )

    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """
        Replace or add an agent in the registry.

        WHY expose this?
          Allows tests to inject mock agents without subclassing or
          monkey-patching. Called in test setUp methods.
        """
        self.agents[name] = agent
        self._log(f"Agent registered: {name} → {type(agent).__name__}")

    def print_trace(self) -> None:
        """Print the complete inter-agent message trace."""
        print("\n─── INTER-AGENT MESSAGE TRACE ───")
        for line in self.message_trace:
            print(f"  {line}")
        print("─────────────────────────────────\n")
