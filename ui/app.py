"""Streamlit front end - one page, four steps: Upload -> Preprocessing -> MCP server -> Investigate.
Talks to the API only (DRI_API_URL, default http://localhost:8000)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import streamlit as st

API_BASE = os.environ.get("DRI_API_URL", "http://localhost:8000")
API = API_BASE + "/api/v1"
PUBLIC_API = os.environ.get("DRI_PUBLIC_API_URL", API_BASE)  # the API address as seen from the user's BROWSER
BADGE = {"contradiction": "🔴 Contradiction", "contextual_difference": "🟡 Contextual difference",
         "insufficient_evidence": "⚪ Insufficient evidence", "consistent": "🟢 Consistent"}

st.set_page_config(page_title="Data Room Investigator", layout="wide")


def call(method: str, path: str, quiet: bool = False, **kw):
    try:
        r = httpx.request(method, API + path, timeout=60, **kw)
    except httpx.HTTPError as e:
        if not quiet:
            st.error(f"API unreachable: {e}")
        return None
    if r.status_code >= 400:
        if not quiet:
            st.error(f"{r.status_code}: {r.json().get('detail', r.text)}")
        return None
    return r.json()


def short(name: str, n: int = 70) -> str:
    return name if len(name) <= n else "…/" + "/".join(name.split("/")[-2:])[-(n - 2):]


def stage(d: dict) -> tuple[str, bool]:
    """(label, is_final) for a document's preprocessing progress."""
    if d["status"] == "failed":
        return "❌ failed", True
    if d["status"] == "uploaded":
        return "⏳ queued", False
    if d["status"] == "processing":
        return "⚙️ parsing & chunking", False
    return {"pending": ("⚙️ summarising", False), "done": ("✅ ready", True),
            "skipped": ("✅ ready (no summary)", True), "failed": ("✅ ready (summary failed)", True)}.get(
        d["summary_status"], ("✅ ready", True))


st.title("Data Room Investigator")

# ------------------------------------------------------------------ 1. Upload
st.header("1 · Upload")
FOLDER_HTML = """
<div style="font-family:system-ui,sans-serif;font-size:14px;color:#444">
  <input id="dir" type="file" webkitdirectory directory multiple>
  <button id="go" disabled style="margin-left:8px">Upload folder</button>
  <pre id="log" style="white-space:pre-wrap;margin-top:8px">Choose a folder: every PDF/TXT/MD/CSV inside it and its subfolders is uploaded and preprocessed automatically.</pre>
</div>
<script>window.__API_URL__ = %s;</script>
<script>%s</script>
"""
_js = (Path(__file__).parent / "folder_upload.js").read_text(encoding="utf-8")
# Our own static HTML + JS (no user-supplied content) - safe for st.iframe, which runs scripts in the page's origin.
st.iframe((FOLDER_HTML % (json.dumps(PUBLIC_API), _js)).strip(), height=150)
files = st.file_uploader("...or pick individual files (PDF, TXT, MD, CSV)", accept_multiple_files=True)
if files and st.button("Upload files"):
    res = call("POST", "/documents", files=[("files", (f.name, f.getvalue())) for f in files]) or []
    bad = [f"{r['filename']}: {r['error']}" for r in res if r.get("error")]
    st.success(f"{sum(1 for r in res if r.get('document'))} file(s) received - preprocessing starts automatically.")
    for b in bad:
        st.warning(b)

# ------------------------------------------------------------------ 2. Preprocessing
st.header("2 · Preprocessing")


@st.fragment(run_every=3)
def preprocessing():
    docs = call("GET", "/documents", quiet=True)
    if docs is None:
        st.info("Waiting for the API...")
        return
    if not docs:
        st.caption("No documents yet.")
        return
    stages = [stage(d) for d in docs]
    ready = sum(1 for d, (_, fin) in zip(docs, stages) if fin and d["status"] == "processed")
    failed = sum(1 for d in docs if d["status"] == "failed")
    st.write(f"**{len(docs)}** documents · **{ready}** ready · **{len(docs) - ready - failed}** in progress · **{failed}** failed")
    st.dataframe([{"File": short(d["filename"]), "Status": s, "Pages": d["page_count"] or "", "Chunks": d["chunk_count"] or "",
                   "Note": d["failure_code"] and f"{d['failure_code']}: {d['error']}" or d["summary_error"] or ""}
                  for d, (s, _) in zip(docs, stages)], hide_index=True, width="stretch")


preprocessing()

# ------------------------------------------------------------------ 3. MCP server
st.header("3 · MCP server")


@st.fragment(run_every=5)
def mcp_panel():
    m = call("GET", "/mcp", quiet=True)
    if m is None:
        st.info("Waiting for the API...")
        return
    if m["ready"]:
        st.success(f"MCP server ready ({m['transport']}) · serving {m['documents_served']} document(s), "
                   f"{m['documents_with_summary']} with summaries · {len(m['tools'])} tools")
    else:
        st.error("MCP server not reachable" + (f": {m['error']}" if m["error"] else ""))
    if m["tools"]:
        st.dataframe([{"Tool": t["name"], "What it does": t["description"]} for t in m["tools"]], hide_index=True, width="stretch")


mcp_panel()

# ------------------------------------------------------------------ 4. Investigate
st.header("4 · Investigate")
docs_now = call("GET", "/documents", quiet=True) or []
busy = [d for d in docs_now if not stage(d)[1]]
usable = [d for d in docs_now if d["status"] == "processed"]
invs = call("GET", "/investigations", quiet=True) or []
inv = max(invs, key=lambda i: i["id"]) if invs else None
running = bool(inv and inv["status"] in ("queued", "running"))

if st.button("Investigate", type="primary", disabled=running or bool(busy) or not usable,
             help="Enabled when every document has finished preprocessing"):
    if call("POST", "/investigations", json={}):
        st.rerun()
if busy:
    st.caption(f"Waiting for preprocessing of {len(busy)} document(s)...")
elif not usable:
    st.caption("Upload documents first.")


@st.fragment(run_every=3 if running else None)
def progress():
    if not inv:
        return
    p = call("GET", f"/investigations/{inv['id']}", quiet=True)
    if not p:
        return
    i = p["investigation"]
    now_running = i["status"] in ("queued", "running")
    st.write(f"**{i['status']}** · {i['phase']} · {p['claims']} claims · "
             f"{p['hypotheses_by_status'].get('done', 0)}/{p['hypotheses_total']} questions investigated · {p['findings']} findings")
    if i["error"]:
        st.error(i["error"])
    if i["status"] in ("interrupted", "failed", "partial") and st.button("Resume / retry failed"):
        call("POST", f"/investigations/{inv['id']}/resume")
        st.rerun()
    events = call("GET", f"/investigations/{inv['id']}/events", quiet=True) or []
    mcp_calls = [e for e in events if e["kind"] == "tool_call"]
    if mcp_calls:
        st.caption("MCP tool calls (latest): " + " · ".join(e["message"] for e in mcp_calls[-5:]))
    with st.expander("Live trace", expanded=now_running):
        st.text("\n".join(f"{e['ts'][11:19]}  {e['kind']:<18} {e['message'][:150]}" for e in reversed(events[-25:])))
    if st.session_state.get("was_running") and not now_running:  # finished: full rerun so findings render
        st.session_state["was_running"] = False
        st.rerun()
    st.session_state["was_running"] = now_running


progress()

if inv:
    fs = call("GET", f"/investigations/{inv['id']}/findings", quiet=True) or []

    def show(f):
        with st.expander(f"{BADGE[f['classification']]} — {f['title']}"):
            st.write(f["explanation"])
            for c in f["citations"]:
                loc = f"{short(c['filename'], 60)} p.{c['page']}" if c["filename"] else f"passage {c['chunk_id']}"
                st.markdown(f"{'✅' if c['verified'] else '❌ not verified'} **{loc}**\n> {c['quote']}")
            for label, key in [("Uncertainties", "uncertainties"), ("Notes", "verification_notes")]:
                if f[key]:
                    st.markdown(f"**{label}**\n" + "\n".join(f"- {x}" for x in f[key]))
            note = st.text_input("Challenge or clarify (optional)", key=f"note{f['id']}")
            if st.button("Investigate this further", key=f"fb{f['id']}"):
                if call("POST", f"/investigations/{inv['id']}/feedback", json={"finding_id": f["id"], "note": note or None}):
                    st.rerun()

    established = [f for f in fs if f["status"] == "established"]
    open_q = [f for f in fs if f["status"] != "established"]
    if established:
        st.subheader(f"Findings ({len(established)})")
        for f in established:
            show(f)
    if open_q:
        st.subheader(f"Unresolved questions ({len(open_q)})")
        for f in open_q:
            show(f)
