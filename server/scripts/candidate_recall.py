"""Map the answer key's cases to the stored claims, then score candidate recall of the key grouping, with no LLM call.

    python scripts/candidate_recall.py [--key KEY.csv] [--set HTC01] [--out test_results/NAME.txt]

1. Case-to-claim map: each citation line of a case ("DOC-05 p.2, 2.3: "$5,120,000"") is matched to the stored claims of that document whose value or sentence holds the
   quoted text (page agreeing scores higher). Citations with no claim are extraction misses. The map is written for a human to confirm.
2. Candidate recall: the key grouping is run again on the stored claims. Every label and the vocabulary merge come from the cache, so nothing is asked of a model
   (a cache miss is reported, not paid for). A case is "connected" when its citations are linked, directly or through others, by claims that share a cluster.
   This is candidate generation only: no gate, no judge.
Reads the dev database; writes one file to test_results.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR))

from sqlalchemy import create_engine, select  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.llm.base import LLMClient, RawResponse  # noqa: E402
from app.llm.errors import LLMPermanentError  # noqa: E402
from app.llm.factory import get_embedding_provider  # noqa: E402
from app.models import Claim, Document, DocumentChunk  # noqa: E402
from app.services.claim_labels import claim_key  # noqa: E402
from app.services.claim_grouping import ordered_rows  # noqa: E402
from app.services.key_grouping import KeyGrouper  # noqa: E402

DEFAULT_KEY = Path.home() / "Downloads" / "AI Data Room Evaluation Data Templates - Ground Truth.csv"
CITATION = re.compile(r'^(DOC-\d+)\s+p\.(\d+)[^:"]*:\s*"(.*)"\s*$')


class NoModel(LLMClient):
    """A client that refuses every call: the labels and the merge must come from the cache."""

    def __init__(self):
        super().__init__(max_attempts=1)
        self.calls = 0

    def _raw_generate(self, prompt: str, system: str | None, json_mode: bool) -> RawResponse:
        self.calls += 1
        raise LLMPermanentError("candidate_recall.py makes no model call")


def squash(text: object) -> str:
    return re.sub(r"[^a-z0-9.%]+", "", str(text or "").lower())


def digits(text: object) -> str:
    return re.sub(r"[^0-9.]", "", str(text or ""))


def parse_citations(cell: str) -> list[tuple[str, int, str]]:
    out = []
    for line in (cell or "").splitlines():
        m = CITATION.match(line.strip())
        if m:
            out.append((m[1], int(m[2]), m[3]))
    return out


def match_claims(claims: list[dict], doc: str, page: int, quote: str) -> list[dict]:
    """The claims of ``doc`` that hold the quote, best first (at most 5). A claim whose own value is the quoted figure or text scores 3; a claim whose sentence
    merely contains the quote scores 1, so the figure's own claim wins over its neighbours in the same sentence. The right page only orders the best tier."""
    q, qd = squash(quote), digits(quote)
    figures = {digits(t) for t in re.findall(r"\d[\d,]*(?:\.\d+)?", quote)}  # every number the quote holds, as digits
    scored = []
    for c in claims:
        if c["doc"] != doc:
            continue
        value, vd = squash(c["value"]), digits(c["value"])
        score = 0
        if q and value and (value == q or (len(value) >= 3 and value in q and not figures)):
            score = 3  # the claim's own value is the quoted text ("March 28")
        elif len(vd) >= 2 and vd in figures:
            score = 3  # the claim's own value is a figure of the quote ("$5,120,000", "84"), whatever its form
        elif q and q in squash(c["text"]):
            score = 1
        elif len(qd) >= 3 and qd in digits(c["text"]):
            score = 1
        if score == 0:
            continue
        scored.append((score, c["page"] == page, c))
    if not scored:
        return []
    best = max(t for t, _p, _c in scored)  # the page only orders the claims of the best tier: the citation's page can be off by one
    return [c for t, _p, c in sorted((x for x in scored if x[0] == best), key=lambda x: not x[1])][:6]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", type=Path, default=DEFAULT_KEY)
    ap.add_argument("--set", default="HTC01")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.out or SERVER_DIR.parent / "test_results" / f"CANDIDATE-RECALL-{stamp}.txt"

    settings = get_settings()
    engine = create_engine(settings.database_url)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()

    rows = list(csv.DictReader(open(args.key, encoding="utf-8-sig")))
    cases = [r for r in rows if r["Case ID"].startswith(args.set)]
    stmt = (select(Claim, Document.filename, DocumentChunk.page_number).join(Document, Document.id == Claim.document_id)
            .join(DocumentChunk, DocumentChunk.id == Claim.chunk_id))
    claims = [dict(id=c.id, doc=f.split("_")[0], value=c.value, text=c.claim_text, page=p, subject=c.subject, metric=c.metric) for c, f, p in session.execute(stmt)]
    by_id = {c["id"]: c for c in claims}

    # ---- 1. the map
    mapping: dict[str, list[dict]] = {}
    for r in cases:
        cites = parse_citations(r["Citations"])
        mapping[r["Case ID"]] = [dict(doc=d, page=p, quote=q, claims=match_claims(claims, d, p, q)) for d, p, q in cites]

    # ---- 2. the grouping, from the cache only
    embedder = get_embedding_provider(settings)
    model = NoModel()
    grouper = KeyGrouper(session, model, factory, embedder=embedder)
    clusters, stats = grouper.run()
    home: dict[str, list[int]] = defaultdict(list)  # claim id -> clusters holding it
    for n, cl in enumerate(clusters):
        for i in cl.claim_ids:
            home[i].append(n)

    lines = [f"CANDIDATE RECALL  {stamp}  (key set {args.set}; {len(claims)} stored claims; {len(clusters)} clusters; model calls made: {model.calls}; "
             f"labels from cache {stats.labelled_cached}, asked {stats.labelled_asked}, fallback {stats.labelled_fallback})"]
    if model.calls or stats.errors:
        lines.append(f"WARNING: a cache miss or an error means this is not the stored run's grouping: {stats.errors[:3]}")
    lines += ["", "=" * 100, "PART 1. CASE-TO-CLAIM MAP (each citation of the key and the stored claims that hold it; confirm by eye)", "=" * 100]
    verdicts = {}
    for r in cases:
        cid = r["Case ID"]
        kind = r["Classification"]
        lines.append(f"\n{cid} [{kind}/{r['Severity']}] {r['Summary']}")
        cites = mapping[cid]
        for c in cites:
            if c["claims"]:
                for m in c["claims"]:
                    lines.append(f"   {c['doc']} p.{c['page']} \"{c['quote'][:60]}\"  ->  {m['id'][-6:]} p.{m['page']} | {str(m['subject'])[:30]} | {str(m['metric'])[:28]} | {str(m['value'])[:40]}")
            else:
                lines.append(f"   {c['doc']} p.{c['page']} \"{c['quote'][:60]}\"  ->  NO CLAIM (not extracted, or worded differently)")

        # connectivity of the citations through shared clusters
        nodes = [i for i, c in enumerate(cites) if c["claims"]]
        edges, routes = set(), set()
        for i, j in combinations(nodes, 2):
            for a in cites[i]["claims"]:
                for b in cites[j]["claims"]:
                    shared = set(home.get(a["id"], [])) & set(home.get(b["id"], []))
                    if shared:
                        edges.add((i, j))
                        for n in shared:
                            routes.update(clusters[n].via.get(a["id"], clusters[n].matched_by) for _ in [0])
        parent = {i: i for i in nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i, j in edges:
            parent[find(i)] = find(j)
        missing = [c for c in cites if not c["claims"]]
        if len(cites) < 2:
            verdict = "n/a (fewer than two citations)"
        elif missing and len(nodes) < 2:
            verdict = "EXTRACTION: too few citations have a claim"
        elif nodes and len({find(i) for i in nodes}) == 1 and not missing:
            verdict = "CONNECTED"
        elif nodes and len({find(i) for i in nodes}) == 1:
            verdict = "CONNECTED (the claims found; some citations have no claim)"
        elif edges:
            verdict = "PARTLY connected"
        else:
            verdict = "NOT connected"
        verdicts[cid] = (kind, verdict, sorted(routes))
        lines.append(f"   => {verdict}" + (f"  (routes: {', '.join(sorted(routes))})" if routes else ""))

    # ---- summary
    issues = [c for c, (k, v, _r) in verdicts.items() if verdicts[c][0] != "Consistent" and not v.startswith("n/a")]
    tally = defaultdict(list)
    for c in issues:
        v = verdicts[c][1]
        tally["connected" if v.startswith("CONNECTED") else "partly" if v.startswith("PARTLY") else "extraction" if v.startswith("EXTRACTION") else "not connected"].append(c)
    lines += ["", "=" * 100, f"PART 2. CANDIDATE RECALL of the key grouping ({len(issues)} non-consistent cases with two or more citations)", "=" * 100]
    for k in ("connected", "partly", "not connected", "extraction"):
        lines.append(f"{k:14s} {len(tally[k]):3d}  {', '.join(c[-3:] for c in tally[k])}")
    lines.append(f"candidate recall (connected / cases) = {len(tally['connected'])}/{len(issues)} = {len(tally['connected']) / max(len(issues), 1):.0%}")
    by_route: dict[str, int] = defaultdict(int)
    for c in tally["connected"] + tally["partly"]:
        for r in verdicts[c][2]:
            by_route[r] += 1
    lines.append(f"cases linked by route (a case can use several): {dict(by_route)}")
    lines.append(f"clusters over 15 claims (the judge splits these): {sum(len(c.claim_ids) > 15 for c in clusters)}; largest {max((len(c.claim_ids) for c in clusters), default=0)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # the map as data, keyed by the claim's own hash (claim ids change every run)
    by_hash = {i: claim_key(v) for i, v in ((r[0].id, r[0]) for r in ordered_rows(session))}
    data = {cid: [dict(doc=c["doc"], page=c["page"], quote=c["quote"], claims=[by_hash.get(m["id"]) for m in c["claims"]]) for c in cites] for cid, cites in mapping.items()}
    out_path.with_suffix(".map.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"wrote {out_path}\nconnected {len(tally['connected'])}/{len(issues)}; partly {len(tally['partly'])}; not connected {len(tally['not connected'])}; extraction {len(tally['extraction'])}; model calls {model.calls}")


if __name__ == "__main__":
    main()
