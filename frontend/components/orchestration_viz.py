"""
Animated HTML/CSS visualization: supervisor orchestrating specialist agents (MCP + LangGraph).
Shown while the workflow API call is in flight.
"""
from __future__ import annotations


def orchestration_html() -> str:
    """Self-contained dark-theme graphic with subtle pulse / flow animation."""
    return """
<div class="orch-wrap">

  <div class="orch-supervisor">
    <span class="orch-badge sup">Supervisor</span>
    <span class="orch-lang">LangGraph</span>
  </div>

  <div class="orch-flow-line" aria-hidden="true"></div>

  <div class="orch-agents">
    <div class="orch-node a1"><span class="orch-ico">◇</span>Discovery</div>
    <div class="orch-node a2"><span class="orch-ico">◇</span>Scoring</div>
    <div class="orch-node a3"><span class="orch-ico">◇</span>Recommend</div>
    <div class="orch-node a4"><span class="orch-ico">◇</span>Outreach</div>
    <div class="orch-node a5"><span class="orch-ico">◇</span>Campaign</div>
  </div>


</div>

<style>
.orch-wrap {
  font-family: ui-sans-serif, system-ui, sans-serif;
  background: linear-gradient(145deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
  border: 1px solid #334155;
  border-radius: 16px;
  padding: 1.25rem 1.5rem 1rem;
  margin: 0.5rem 0 1rem 0;
  box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.orch-title {
  font-size: 1.05rem;
  font-weight: 700;
  color: #f1f5f9;
  letter-spacing: -0.02em;
}
.orch-sub {
  font-size: 0.78rem;
  color: #94a3b8;
  margin-top: 0.35rem;
  margin-bottom: 1.1rem;
  line-height: 1.4;
}
.orch-supervisor {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.6rem;
  flex-wrap: wrap;
}
.orch-badge {
  display: inline-flex;
  align-items: center;
  padding: 0.45rem 1rem;
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 600;
}
.orch-badge.sup {
  background: linear-gradient(135deg, #ea580c 0%, #c2410c 100%);
  color: #fff;
  box-shadow: 0 0 20px rgba(234, 88, 12, 0.35);
  animation: orch-glow 2s ease-in-out infinite;
}
.orch-lang {
  font-size: 0.72rem;
  color: #64748b;
  border: 1px solid #475569;
  padding: 0.25rem 0.55rem;
  border-radius: 6px;
}
.orch-flow-line {
  height: 3px;
  margin: 0.85rem auto 0.75rem;
  max-width: 280px;
  border-radius: 3px;
  background: linear-gradient(90deg, transparent, #38bdf8, #a78bfa, #34d399, transparent);
  background-size: 200% 100%;
  animation: orch-shine 2.2s linear infinite;
}
.orch-agents {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 0.45rem;
  margin-top: 0.15rem;
}
.orch-node {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.38rem 0.65rem;
  border-radius: 10px;
  font-size: 0.72rem;
  font-weight: 600;
  color: #e2e8f0;
  background: #1e293b;
  border: 1px solid #475569;
  animation: orch-pulse 1.8s ease-in-out infinite;
}
.orch-node .orch-ico { font-size: 0.55rem; opacity: 0.85; }
.orch-node.a1 { animation-delay: 0s; }
.orch-node.a2 { animation-delay: 0.15s; }
.orch-node.a3 { animation-delay: 0.3s; }
.orch-node.a4 { animation-delay: 0.45s; }
.orch-node.a5 { animation-delay: 0.6s; }
.orch-footer {
  margin-top: 1rem;
  font-size: 0.7rem;
  color: #64748b;
  text-align: center;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.35rem;
}
.orch-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #22c55e;
  animation: orch-blink 1s ease-in-out infinite;
}
@keyframes orch-pulse {
  0%, 100% { opacity: 0.75; transform: translateY(0); box-shadow: none; }
  50% { opacity: 1; transform: translateY(-2px); box-shadow: 0 4px 14px rgba(56, 189, 248, 0.2); }
}
@keyframes orch-shine {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
@keyframes orch-glow {
  0%, 100% { filter: brightness(1); }
  50% { filter: brightness(1.12); }
}
@keyframes orch-blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.35; }
}
</style>
"""
