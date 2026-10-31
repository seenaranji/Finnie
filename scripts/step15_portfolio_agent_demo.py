"""Step 15: the Portfolio Analysis agent on the three sample portfolios (real prices, real model).

Run:  .venv/bin/python -m scripts.step15_portfolio_agent_demo
"""
import json
from datetime import datetime
from pathlib import Path

from src.agents.portfolio_agent import PortfolioAnalysisAgent
from src.core.config import PROJECT_ROOT
from src.core.llm import LLMClient
from src.core.logging import setup_logging
from src.data.factory import build_market_data_service

setup_logging("WARNING")
metadata = json.loads((PROJECT_ROOT / "src/data/ticker_metadata.json").read_text())
agent = PortfolioAnalysisAgent(LLMClient(), build_market_data_service(), metadata)
CASES = [("diversified", None), ("concentrated_single_stock", None), ("bond_heavy", None),
         ("concentrated_single_stock", "Should I sell some of my Apple stock?")]
rows = []
for name, question in CASES:
    csv_text = (PROJECT_ROOT / f"src/data/sample_portfolios/{name}.csv").read_text()
    response = agent.run(csv_text, question) if question else agent.run(csv_text)
    meta = response.metadata
    print("=" * 100, f"\n{name}  |  question: {question or '(default: explain my portfolio)'}\n" + "=" * 100)
    print(response.text)
    print(f"\n  sources: {[s.title for s in response.sources]}")
    print(f"  advice_requested={meta['advice_requested']}  output_flagged={meta['output_flagged']}")
    print(f"  disclaimer: {response.disclaimer}")
    print(f"  time: data fetch {meta['data_seconds']:.2f}s, math {meta['calc_seconds'] * 1000:.1f}ms, LLM {meta['llm_seconds']:.2f}s, total {meta['latency_s']:.2f}s; "
          f"tokens {meta['input_tokens']}/{meta['output_tokens']}; cost ${meta['cost_usd']:.5f}")
    rows.append({"portfolio": name, "question": question, "text": response.text, "metadata": meta})

print("\n" + "=" * 100, "\nWHAT THE MODEL WAS GIVEN (METRICS for the concentrated portfolio)\n" + "=" * 100)
print(json.dumps(rows[1]["metadata"]["metrics"], indent=1))
output_file = Path(f"benchmarks/results/step15_portfolio_agent_demo_{datetime.now():%Y%m%d_%H%M%S}.json")
output_file.write_text(json.dumps(rows, indent=2, default=str))
print(f"\nsaved: {output_file}")
