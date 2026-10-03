"""
agents/message.py
─────────────────
Defines the AgentMessage dataclass — the single typed communication
contract used across all agents in this system.

WHY a dataclass rather than a plain dict?
  A dict would allow any key to be passed silently. Using a dataclass
  enforces the message schema at construction time, meaning a malformed
  message causes an immediate TypeError rather than a silent downstream
  failure. This was the foundational design decision that made the
  unit-test suite reliable (Wooldridge, 2009, p.6 stresses that MAS
  reliability depends on well-defined agent interfaces).

WHY ISO 8601 UTC for timestamps?
  REMEDIATION #3: early versions used datetime.now() which produced
  local-time strings, causing log sequence analysis to break when the
  system was run in different time zones (e.g., UAE vs. UTC server).
  Standardised to UTC so log traces are unambiguous regardless of
  runtime environment.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid


@dataclass
class AgentMessage:
    """
    Immutable typed envelope for all inter-agent communication.

    Attributes
    ----------
    sender      : Name of the originating agent (e.g. 'orchestrator').
    recipient   : Name of the target agent (e.g. 'planner').
    payload     : Structured dict carrying task-specific data.
    message_id  : UUID4 auto-generated for traceability.
    timestamp   : ISO 8601 UTC string — always UTC, never local time.
    """

    sender: str
    recipient: str
    payload: dict

    # Auto-generated fields — callers never need to supply these.
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def summary(self) -> str:
        """One-line log summary for the execution trace."""
        keys = list(self.payload.keys())
        return (
            f"[MSG] {self.sender}→{self.recipient} "
            f"| keys={keys} | id={self.message_id[:8]}"
        )
