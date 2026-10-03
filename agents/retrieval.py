"""
agents/retrieval.py
───────────────────
The RetrievalAgent performs semantic similarity search over a local
FAISS vector index of telecom fault knowledge-base entries.

WHY FAISS + sentence-transformers rather than a cloud vector DB?
  Three reasons:
    1. Cost: FAISS runs locally with no API key or subscription.
    2. Portability: the index is built once and saved to disk —
       any machine with the repo can reproduce results exactly.
    3. Testability: a fixed KB produces deterministic retrieval
       results, which is essential for unit testing.
  (Lewis et al., 2020 establish RAG as the standard pattern for
  grounding LLM outputs in domain knowledge.)

WHY TOP_K = 3?
  Empirically tested during development:
    K=1 → too narrow; misses complementary fault patterns.
    K=5 → introduces low-relevance noise into the LLM prompt,
          increasing token count without improving output quality.
    K=3 → best balance between context richness and prompt budget.
  This decision is documented here so it is not treated as magic.

REMEDIATION #2 — FAISS Index Persistence:
  Initial version rebuilt the index on every run (~4 s startup overhead).
  Fixed by: saving index + KB hash to disk after first build; loading
  from disk on subsequent runs if the KB content hash matches.
  Startup time reduced from ~4 s to <0.2 s.
"""

import os
import json
import hashlib
import numpy as np

from agents.base import BaseAgent
from agents.message import AgentMessage

# ── Constants ─────────────────────────────────────────────────────────────
TOP_K = 3

# FAISS_INDEX_DIR can be overridden via environment variable.
# Default : kb/           (local and Colab — writable working directory)
# Spaces  : /tmp/faiss_cache (set by app.py — always writable on Spaces)
# WHY: HuggingFace Spaces has a read-only app filesystem at runtime.
# Redirecting to /tmp ensures the index builds and caches without a
# PermissionError on first run.
_FAISS_DIR = os.environ.get("FAISS_INDEX_DIR", "kb")
INDEX_PATH = os.path.join(_FAISS_DIR, "faiss.index")
META_PATH  = os.path.join(_FAISS_DIR, "faiss_meta.json")

# ── Knowledge Base ─────────────────────────────────────────────────────────
# 15 telecom fault entries drawn from ITU-T G.7710 fault taxonomy and
# the author's NOC operational experience.
KNOWLEDGE_BASE = [
    {
        "id": "KB-001",
        "title": "BGP Session Drop — Keepalive Timer Mismatch",
        "content": "BGP session instability caused by keepalive timer mismatch between peers. "
                   "PE router configured at 30s, CE router at 60s. Session flaps every ~90s.",
        "resolution": "Align keepalive timers on both peers. Standard: 60s hold-time, 20s keepalive.",
        "tags": ["BGP", "MPLS", "PE", "session", "timer"],
    },
    {
        "id": "KB-002",
        "title": "MPLS Label Stack Overflow",
        "content": "MPLS label stack depth exceeded router hardware limit of 3 labels. "
                   "Occurs on inter-AS MPLS VPN topologies with recursive next-hop resolution.",
        "resolution": "Reduce label stack depth. Enable MPLS TE shortcuts or adjust recursive lookup policy.",
        "tags": ["MPLS", "label", "stack", "inter-AS", "VPN"],
    },
    {
        "id": "KB-003",
        "title": "Interface MTU Mismatch — Path MTU Black Hole",
        "content": "Asymmetric MTU configuration causes Path MTU Discovery black hole. "
                   "Customer traffic silently dropped for packets >1500B. Affects MPLS P-links.",
        "resolution": "Set consistent MTU across all P-links (recommended: 1600B for MPLS transport). "
                      "Enable 'ip tcp adjust-mss 1452' on CE-facing interfaces.",
        "tags": ["MTU", "MPLS", "blackhole", "interface", "TCP"],
    },
    {
        "id": "KB-004",
        "title": "BGP Route Leak — Missing Outbound Filter",
        "content": "Customer prefixes leaked to transit peers due to missing outbound route filter. "
                   "Caused by misconfigured route-map applied only to IPv4, missing IPv6.",
        "resolution": "Apply prefix-list filter to all AFI/SAFI. Validate with 'show bgp neighbor advertised-routes'.",
        "tags": ["BGP", "route-leak", "filter", "prefix-list", "IPv6"],
    },
    {
        "id": "KB-005",
        "title": "Fibre Cut — Single-mode Span Break",
        "content": "Physical break on single-mode fibre span. OTDR reading shows 40 dB loss at 12.4 km "
                   "from DXB POP. Likely cable dig-up on E311 highway section.",
        "resolution": "Dispatch field team for splice repair. Activate 1+1 protection on affected span. "
                      "ETA repair: 4-6 hours. Notify customers of P1 outage.",
        "tags": ["fibre", "physical", "OTDR", "span", "protection"],
    },
    {
        "id": "KB-006",
        "title": "PE Router CPU Spike — BGP Scanner Process",
        "content": "PE router CPU exceeding 90% due to BGP scanner process. Triggered by full BGP "
                   "table import from transit peer (800k+ prefixes) without route summarisation.",
        "resolution": "Apply inbound prefix filter to transit session. Enable route summarisation. "
                      "Upgrade route processor memory if persistent.",
        "tags": ["BGP", "CPU", "PE", "scanner", "performance"],
    },
    {
        "id": "KB-007",
        "title": "OSPF Neighbour Flap — BFD Misconfiguration",
        "content": "OSPF neighbour relationship flapping due to BFD timer set below hardware minimum "
                   "(50ms configured, hardware minimum 100ms). BFD falsely declares peer dead.",
        "resolution": "Increase BFD timers to ≥300ms. Verify hardware BFD support on line card.",
        "tags": ["OSPF", "BFD", "neighbour", "timer", "flap"],
    },
    {
        "id": "KB-008",
        "title": "VLAN Misconfiguration — Trunk Port Tag Mismatch",
        "content": "Customer VLAN not forwarded across trunk port. Access switch sending tagged frames "
                   "on VLAN 200, aggregation switch expects native VLAN 200 (untagged). Frame discarded.",
        "resolution": "Align VLAN tagging mode on both trunk endpoints. Document expected VLAN topology.",
        "tags": ["VLAN", "trunk", "tag", "Ethernet", "switching"],
    },
    {
        "id": "KB-009",
        "title": "DNS Resolution Failure — Resolver Misconfiguration",
        "content": "Customer reporting intermittent name resolution failures. Root cause: resolver "
                   "pointing to decommissioned secondary DNS server 10.10.10.2.",
        "resolution": "Update resolver configuration to active servers. Implement DNS monitoring.",
        "tags": ["DNS", "resolver", "configuration", "intermittent"],
    },
    {
        "id": "KB-010",
        "title": "QoS Policy Drop — DSCP Remarking Error",
        "content": "Voice traffic experiencing packet loss. DSCP EF (46) traffic being remarked to "
                   "Best Effort (0) at customer handoff interface due to absent trust boundary config.",
        "resolution": "Apply 'mls qos trust dscp' on customer-facing interfaces. Audit QoS policy chain.",
        "tags": ["QoS", "DSCP", "voice", "VoIP", "remarking"],
    },
    {
        "id": "KB-011",
        "title": "MPLS VPN Route Target Mismatch",
        "content": "Customer sites unable to communicate within MPLS L3VPN. Import/export route targets "
                   "mismatched: Site A exports RT 65001:100, Site B imports RT 65001:200.",
        "resolution": "Align VRF route-target configuration across all PE routers serving the VPN.",
        "tags": ["MPLS", "VPN", "VRF", "route-target", "L3VPN"],
    },
    {
        "id": "KB-012",
        "title": "Link Aggregation — LACP Negotiation Failure",
        "content": "Aggregated link operating as single member only. LACP timeout mode mismatch: "
                   "one end configured 'fast' (1s PDU), other end configured 'slow' (30s PDU).",
        "resolution": "Align LACP timeout mode to 'fast' on both ends for quicker failover detection.",
        "tags": ["LACP", "LAG", "aggregation", "bonding", "Ethernet"],
    },
    {
        "id": "KB-013",
        "title": "Optical Amplifier Gain Tilt — DWDM Degradation",
        "content": "Multiple DWDM channels experiencing elevated BER. EDFA gain tilt detected "
                   "following maintenance window. Channels at band edges showing -3 dB power differential.",
        "resolution": "Re-commission EDFA gain equalisation. Check VOA settings per channel. "
                      "Perform OSNR measurement sweep.",
        "tags": ["DWDM", "EDFA", "optical", "BER", "amplifier", "fibre"],
    },
    {
        "id": "KB-014",
        "title": "Routing Loop — Redistribution Between OSPF and BGP",
        "content": "Routing loop causing packet TTL expiry. Static route redistributed into OSPF, "
                   "then redistributed into BGP and back into OSPF. Administrative distance conflict.",
        "resolution": "Apply route-map with tag to prevent re-redistribution. Review redistribution "
                      "policy comprehensively.",
        "tags": ["routing", "loop", "redistribution", "OSPF", "BGP", "AD"],
    },
    {
        "id": "KB-015",
        "title": "Segment Routing — MPLS Forwarding Plane Failure",
        "content": "SR-MPLS traffic not forwarding after IOS-XR upgrade. SRGB (Segment Routing "
                   "Global Block) not re-advertised after software reload. FIB entries stale.",
        "resolution": "Clear MPLS forwarding table and trigger SR re-advertisement. "
                      "Verify SRGB range consistency across all SR nodes.",
        "tags": ["segment-routing", "SR-MPLS", "SRGB", "FIB", "IOS-XR"],
    },
]


def _kb_hash() -> str:
    """Compute a hash of the KB content for cache invalidation."""
    content = json.dumps(KNOWLEDGE_BASE, sort_keys=True).encode()
    return hashlib.md5(content).hexdigest()


class RetrievalAgent(BaseAgent):
    """
    Semantic retrieval agent using FAISS + sentence-transformers.

    On first call: encodes KB entries, builds FAISS index, persists to disk.
    On subsequent calls: loads index from disk if KB hash matches.

    Input payload keys:  query (str), incident_type (str)
    Output payload keys: retrieved_docs (list[dict]), query (str)
    """

    def __init__(self):
        super().__init__("retrieval")
        self._model = None
        self._index = None
        self._kb_texts = [entry["content"] for entry in KNOWLEDGE_BASE]

    def _load_model(self):
        """Lazy-load the embedding model — avoids import-time cost."""
        if self._model is None:
            self._log("Loading sentence-transformer model (all-MiniLM-L6-v2)...")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._log("Model loaded.")

    def _build_or_load_index(self):
        """
        Build or load the FAISS index.

        REMEDIATION #2: previously rebuilt on every run (~4s).
        Now persists index + KB hash to disk; reloads if hash matches.
        """
        import faiss

        if self._index is not None:
            return   # already loaded this session

        current_hash = _kb_hash()

        # Try to load from disk
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            with open(META_PATH, "r") as f:
                meta = json.load(f)
            if meta.get("kb_hash") == current_hash:
                self._log("Loading FAISS index from disk (KB unchanged).")
                self._index = faiss.read_index(INDEX_PATH)
                self._log(f"Index loaded: {self._index.ntotal} vectors.")
                return

        # Build fresh index
        self._log("Building FAISS index from knowledge base...")
        self._load_model()
        embeddings = self._model.encode(self._kb_texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)   # cosine similarity via inner product

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)   # Inner Product = cosine after L2 norm
        self._index.add(embeddings)

        # Persist
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        faiss.write_index(self._index, INDEX_PATH)
        with open(META_PATH, "w") as f:
            json.dump({"kb_hash": current_hash, "n_entries": len(KNOWLEDGE_BASE)}, f)
        self._log(f"Index built and saved: {self._index.ntotal} vectors, dim={dim}.")

    def run(self, message: AgentMessage) -> AgentMessage:
        query = message.payload.get("query", "")
        self._log(f"Query: '{query}'")

        self._build_or_load_index()
        self._load_model()

        # Encode and normalise query
        import faiss, numpy as np
        q_vec = self._model.encode([query], show_progress_bar=False)
        q_vec = np.array(q_vec, dtype="float32")
        faiss.normalize_L2(q_vec)

        # Search
        scores, indices = self._index.search(q_vec, TOP_K)

        retrieved = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            entry = KNOWLEDGE_BASE[idx].copy()
            entry["similarity"] = round(float(score), 4)
            entry["rank"] = rank + 1
            retrieved.append(entry)
            self._log(
                f"  Rank {rank+1}: [{entry['id']}] {entry['title']} "
                f"(sim={entry['similarity']:.4f})"
            )

        return AgentMessage(
            sender=self.name,
            recipient="orchestrator",
            payload={
                "retrieved_docs": retrieved,
                "query": query,
                "n_retrieved": len(retrieved),
            },
        )
