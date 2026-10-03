"""
agents/base.py
──────────────
Abstract base class for all agents in the multi-agent system.

WHY an abstract base class?
  All agents share the same interface: they receive an AgentMessage
  and return an AgentMessage. Enforcing this via ABC means any agent
  can be swapped in the orchestrator's registry without changing the
  dispatch logic — the Liskov Substitution Principle applied to agents
  (Russell and Norvig, 2021, p.55 describe this as the key property of
  a well-defined agent architecture).

WHY store a log?
  Every agent accumulates its own execution log as a list of strings.
  The orchestrator collects these after each run, giving a complete
  per-agent trace that is used in functional and integration tests.
"""

from abc import ABC, abstractmethod
from agents.message import AgentMessage


class BaseAgent(ABC):
    """
    Abstract base for all system agents.

    Subclasses must implement `run(message)`.
    All agents share a self.log list for execution tracing.
    """

    def __init__(self, name: str):
        self.name = name
        self.log: list[str] = []

    @abstractmethod
    def run(self, message: AgentMessage) -> AgentMessage:
        """
        Process an incoming message and return a response message.

        Parameters
        ----------
        message : AgentMessage
            Typed envelope from the orchestrator or upstream agent.

        Returns
        -------
        AgentMessage
            Typed envelope to be routed onward by the orchestrator.
        """
        ...

    def _log(self, entry: str) -> None:
        """Append a timestamped entry to this agent's execution log."""
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        line = f"[{ts}] [{self.name.upper()}] {entry}"
        self.log.append(line)
        print(line)   # visible in Colab cell output
