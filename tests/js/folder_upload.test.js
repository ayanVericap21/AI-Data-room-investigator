// Run: node --test tests/js     (Node >= 20: File / FormData / fetch are built in)
const test = require("node:test");
const assert = require("node:assert");
const { classify, uploadFiles, relPath } = require("../../ui/folder_upload.js");

function file(rel, content = "x") {
  const f = new File([content], rel.split("/").pop());
  Object.defineProperty(f, "webkitRelativePath", { value: rel.includes("/") ? rel : "" });
  return f;
}

test("classify keeps supported files from subfolders and skips the rest", () => {
  const picked = [
    file("room/a.pdf"), file("room/sub/b.TXT"), file("room/sub/deep/c.md"), file("room/d.csv"),
    file("room/e.docx"), file("room/f.png"), file("room/noext"),
    file("room/.git/config.txt"), file("room/.hidden.txt"), file("room/node_modules/x/readme.md"),
  ];
  const { accepted, skipped } = classify(picked);
  assert.deepStrictEqual(accepted.map(relPath), ["room/a.pdf", "room/sub/b.TXT", "room/sub/deep/c.md", "room/d.csv"]);
  assert.deepStrictEqual(skipped, { hidden: 3, unsupported: 3 });
});

test("classify handles files picked without a folder (no relative path)", () => {
  const { accepted } = classify([file("plain.txt")]);
  assert.strictEqual(relPath(accepted[0]), "plain.txt");
});

test("uploadFiles sends relative paths, in batches, and tallies results", async () => {
  const calls = [];
  const fetchImpl = async (url, init) => {
    const names = init.body.getAll("files").map((f) => f.name);
    calls.push({ url, names });
    return { ok: true, json: async () => names.map((n) => (n.endsWith("dup.txt") ? { filename: n, duplicate: true, document: {} }
                                                         : n.endsWith("bad.txt") ? { filename: n, error: "file is empty" }
                                                         : { filename: n, duplicate: false, document: {} })) };
  };
  const files = ["r/a.txt", "r/s/b.txt", "r/s/dup.txt", "r/s/bad.txt", "r/t/c.txt"].map((p) => file(p));
  const progress = [];
  const out = await uploadFiles(files, "http://api.test", (d, n) => progress.push([d, n]), fetchImpl, 2);
  assert.deepStrictEqual(calls.map((c) => c.names), [["r/a.txt", "r/s/b.txt"], ["r/s/dup.txt", "r/s/bad.txt"], ["r/t/c.txt"]]);
  assert.strictEqual(calls[0].url, "http://api.test/api/v1/documents");
  assert.deepStrictEqual(out, { added: 3, duplicate: 1, errors: [{ file: "r/s/bad.txt", error: "file is empty" }] });
  assert.deepStrictEqual(progress, [[2, 5], [4, 5], [5, 5]]);
});

test("uploadFiles reports HTTP and network failures per file without aborting", async () => {
  let n = 0;
  const fetchImpl = async () => {
    n++;
    if (n === 1) return { ok: false, status: 503 };
    throw new Error("connection refused");
  };
  const out = await uploadFiles([file("r/a.txt"), file("r/b.txt"), file("r/c.txt")], "http://x", null, fetchImpl, 2);
  assert.strictEqual(out.added, 0);
  assert.deepStrictEqual(out.errors.map((e) => e.error), ["HTTP 503", "HTTP 503", "network error: connection refused"]);
});
