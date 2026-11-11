// Folder upload for the Streamlit UI: picks a folder (with subfolders) and POSTs the supported files to the API,
// preserving each file's relative path (e.g. "data-room/legal/lease.pdf"). Pure logic is exported for Node tests.
const SUPPORTED = [".pdf", ".txt", ".md", ".csv"];
const SKIP_DIRS = ["node_modules", "__pycache__"];

function relPath(f) { return f.webkitRelativePath || f.name; }

// Split picked files into uploadable ones and skipped ones (hidden files/folders, unsupported types).
function classify(files) {
  const accepted = [];
  const skipped = { hidden: 0, unsupported: 0 };
  for (const f of files) {
    const segs = relPath(f).split("/");
    const last = segs[segs.length - 1].toLowerCase();
    const ext = last.includes(".") ? last.slice(last.lastIndexOf(".")) : "";
    if (segs.some((s) => s.startsWith(".") || SKIP_DIRS.includes(s))) skipped.hidden++;
    else if (!SUPPORTED.includes(ext)) skipped.unsupported++;
    else accepted.push(f);
  }
  return { accepted, skipped };
}

// Upload in small batches so one request never carries the whole data room.
async function uploadFiles(files, api, onProgress, fetchImpl, batchSize) {
  const doFetch = fetchImpl || fetch;
  const size = batchSize || 10;
  const out = { added: 0, duplicate: 0, errors: [] };
  for (let i = 0; i < files.length; i += size) {
    const batch = files.slice(i, i + size);
    const fd = new FormData();
    for (const f of batch) fd.append("files", f, relPath(f)); // 3rd arg = the relative path the server keeps
    try {
      const r = await doFetch(api + "/api/v1/documents", { method: "POST", body: fd });
      if (!r.ok) {
        for (const f of batch) out.errors.push({ file: relPath(f), error: "HTTP " + r.status });
      } else {
        for (const item of await r.json()) {
          if (item.error) out.errors.push({ file: item.filename, error: item.error });
          else if (item.duplicate) out.duplicate++;
          else out.added++;
        }
      }
    } catch (e) {
      for (const f of batch) out.errors.push({ file: relPath(f), error: "network error: " + e.message });
    }
    if (onProgress) onProgress(Math.min(i + size, files.length), files.length);
  }
  return out;
}

if (typeof module !== "undefined") module.exports = { classify, uploadFiles, relPath };

// ---- DOM binding (browser only) ----
if (typeof document !== "undefined" && document.getElementById("dir")) {
  const API = window.__API_URL__;
  const dir = document.getElementById("dir"), go = document.getElementById("go"), log = document.getElementById("log");
  let accepted = [];
  dir.onchange = () => {
    const c = classify(Array.from(dir.files));
    accepted = c.accepted;
    const folders = new Set(accepted.map((f) => relPath(f).split("/").slice(0, -1).join("/"))).size;
    log.textContent = `${accepted.length} file(s) in ${folders} folder(s) ready` +
      (c.skipped.unsupported ? `, ${c.skipped.unsupported} skipped (unsupported type)` : "") +
      (c.skipped.hidden ? `, ${c.skipped.hidden} skipped (hidden/system folders)` : "");
    go.disabled = accepted.length === 0;
  };
  go.onclick = async () => {
    go.disabled = true; dir.disabled = true;
    const res = await uploadFiles(accepted, API, (d, n) => { log.textContent = `Uploading ${d}/${n} ...`; });
    log.textContent = `Done: ${res.added} added, ${res.duplicate} already present, ${res.errors.length} failed.` +
      (res.errors.length ? "\n" + res.errors.slice(0, 20).map((e) => ` - ${e.file}: ${e.error}`).join("\n") : "") +
      "\nPreprocessing started automatically - see step 2 below.";
    dir.disabled = false; dir.value = ""; accepted = [];
  };
}
