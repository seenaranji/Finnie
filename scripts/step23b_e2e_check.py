"""Step 23b: use the running API end to end against PostgreSQL: save a portfolio, load it by id, analyze it by id and from CSV (the metrics must match),
edit it, delete it, and try the error paths. Costs two portfolio explanations (about $0.01) and uses live market prices.

Start the server first:  .venv/bin/uvicorn src.web_app.api:app --port 8766
Run:  .venv/bin/python -m scripts.step23b_e2e_check [http://127.0.0.1:8766]
"""
import json
import sys
import time

import httpx

from src.core.config import PROJECT_ROOT

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8766"
CSV = (PROJECT_ROOT / "src/data/sample_portfolios/diversified.csv").read_text()
HOLDINGS = [{"ticker": t, "shares": s, "avg_cost": c} for t, s, c in (line.split(",") for line in CSV.strip().splitlines()[1:])]
failures = []


def check(label, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {label} {detail}")
    if not ok:
        failures.append(label)


with httpx.Client(base_url=base, timeout=60) as client:
    # since Step 27 every endpoint needs a login: register a throwaway user for this check (random email, so re-running never clashes)
    import uuid
    email = f"e2e-{uuid.uuid4().hex[:8]}@example.com"
    signup = client.post("/auth/register", json={"email": email, "password": "e2e-check-password-1"})
    check("POST /auth/register -> 201 with a token", signup.status_code == 201, email)
    client.headers["Authorization"] = f"Bearer {signup.json()['access_token']}"
    check("an anonymous request is refused", httpx.get(f"{base}/portfolios", timeout=10).status_code == 401)
    health = client.get("/health").json()
    print("health:", health)

    created = client.post("/portfolios", json={"name": "e2e diversified", "holdings": HOLDINGS})
    pid = created.json()["id"]
    check("POST /portfolios -> 201 with an id", created.status_code == 201 and len(pid) == 36, f"id={pid}")

    got = client.get(f"/portfolios/{pid}").json()
    check("GET /portfolios/{id} returns the same holdings", [(h["ticker"], float(h["shares"]), float(h["avg_cost"])) for h in got["holdings"]]
          == [(h["ticker"], float(h["shares"]), float(h["avg_cost"])) for h in HOLDINGS])
    check("the saved portfolio is in the list", pid in [p["id"] for p in client.get("/portfolios").json()])

    start = time.perf_counter()
    by_id = client.get(f"/portfolios/{pid}/analysis")
    by_id_s = time.perf_counter() - start
    start = time.perf_counter()
    by_csv = client.post("/portfolio/analyze", json={"csv_text": CSV})
    by_csv_s = time.perf_counter() - start
    check("analysis by id = 200", by_id.status_code == 200, f"({by_id_s:.2f} s; CSV route {by_csv_s:.2f} s)")
    a, b = by_id.json()["metrics"], by_csv.json()["metrics"]
    check("metrics by id == metrics from the CSV (same prices_as_of, weights, HHI, volatility ...)", a == b,
          "" if a == b else f"differs in: {[k for k in a if a[k] != b.get(k)]}")
    print("   holdings:", [(h["ticker"], h["weight_pct"]) for h in a["holdings"]], "HHI", a["hhi_concentration_0_to_1"], "prices as of", a["prices_as_of"])

    check("PUT replaces holdings", client.put(f"/portfolios/{pid}", json={"name": "e2e edited", "holdings": HOLDINGS[:2]}).status_code == 204
          and len(client.get(f"/portfolios/{pid}").json()["holdings"]) == 2 and client.get(f"/portfolios/{pid}").json()["name"] == "e2e edited")

    missing = "00000000-0000-4000-8000-000000000000"
    r = client.get(f"/portfolios/{missing}")
    check("missing id -> 404 portfolio_not_found", r.status_code == 404 and r.json()["code"] == "portfolio_not_found", json.dumps(r.json()))
    r = client.get(f"/portfolios/{missing}/analysis")
    check("analysis of a missing id -> 404", r.status_code == 404)
    r = client.post("/portfolios", json={"name": "bad", "holdings": [{"ticker": "", "shares": "-3", "avg_cost": "1"}]})
    check("negative shares and empty ticker -> 422 naming the fields", r.status_code == 422 and {"holdings.0.ticker", "holdings.0.shares"} <= {d["field"] for d in r.json()["details"]},
          json.dumps(r.json()["details"]))
    check("malformed id -> 422", client.get("/portfolios/not-a-uuid").status_code == 422)

    # a second user must not be able to see, change, analyze or delete the first user's portfolio (Step 27)
    other = httpx.post(f"{base}/auth/register", json={"email": f"e2e-{uuid.uuid4().hex[:8]}@example.com", "password": "e2e-check-password-2"}, timeout=30).json()
    bob = {"Authorization": f"Bearer {other['access_token']}"}
    bob_view = [client.get(f"/portfolios/{pid}", headers=bob).status_code, client.get(f"/portfolios/{pid}/analysis", headers=bob).status_code,
                client.put(f"/portfolios/{pid}", json={"name": "x", "holdings": HOLDINGS[:1]}, headers=bob).status_code, client.delete(f"/portfolios/{pid}", headers=bob).status_code]
    check("another user gets 404 for get / analysis / put / delete of my portfolio", bob_view == [404, 404, 404, 404], str(bob_view))
    check("another user's list does not show it", client.get("/portfolios", headers=bob).json() == [])
    check("my portfolio is still intact after those attempts", client.get(f"/portfolios/{pid}").json()["name"] == "e2e edited")

    # a portfolio typed in with a kind per holding, shown grouped (live prices; no model call)
    typed = client.post("/portfolios", json={"name": "e2e typed", "holdings": [
        {"ticker": "VTI", "asset_type": "etf", "shares": "10", "avg_cost": "200.50", "purchase_date": "2024-02-01"},
        {"ticker": "AAPL", "asset_type": "stock", "shares": "5", "avg_cost": "150"},
        {"ticker": "BND", "asset_type": "bond", "shares": "20", "avg_cost": "72"},
        {"ticker": "CASH", "asset_type": "cash", "shares": "1000", "avg_cost": "1"}]})
    typed_id = typed.json()["id"]
    check("typed portfolio saved with its own unique id", typed.status_code == 201 and typed_id != pid, typed_id)
    summary = client.get(f"/portfolios/{typed_id}/summary").json()
    groups = {g["asset_type"]: g for g in summary["groups"]}
    check("grouped by the kinds I entered", set(groups) == {"etf", "stock", "bond", "cash"}, str(sorted(groups)))
    check("cash is worth exactly its face value", groups["cash"]["value"] == 1000.0 and groups["cash"]["gain_pct"] == 0.0)
    check("total cost is exact: 10*200.50 + 5*150 + 20*72 + 1000", summary["total_cost"] == 2005 + 750 + 1440 + 1000, str(summary["total_cost"]))
    check("group values add up to the total value", abs(sum(g["value"] for g in summary["groups"]) - summary["total_value"]) < 0.05, f"{summary['total_value']}")
    check("weights of the groups add up to 100%", abs(sum(g["weight_pct"] or 0 for g in summary["groups"]) - 100) < 0.1)
    print("   ", [(g["label"], g["value"], g["weight_pct"]) for g in summary["groups"]], "unpriced:", summary["unpriced"])
    check("another user cannot read it", client.get(f"/portfolios/{typed_id}/summary", headers=bob).status_code == 404)
    client.delete(f"/portfolios/{typed_id}")

    check("DELETE -> 204, then 404", client.delete(f"/portfolios/{pid}").status_code == 204 and client.get(f"/portfolios/{pid}").status_code == 404)

    chat = client.post("/chat", json={"message": "What is an expense ratio?"}).json()
    check("chat answers from the knowledge base (retrieval on PostgreSQL)", chat["answered_by"] == "finance_qa" and bool(chat["sources"]), f"sources={[s['id'] for s in chat['sources']]}")

print("\nfailures:", failures or "none")
print("\n*** E2E CHECK FINISHED ***")
