import { useEffect, useMemo, useRef, useState } from "react";
import { Alert, Box, Button, Chip, CircularProgress, MenuItem, Paper, Stack, TextField, Typography } from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import { errorMessage } from "../api/client";
import FileTree from "../components/common/FileTree";
import { POLL_MS, useRunFiles, useRunLog, useStartRun, useStopRun } from "../hooks/useTestRun";
import type { RunState } from "../types/testRun";
import { allFolders, buildTree } from "../utils/fileTree";

const STATE_COLOR: Record<RunState, "default" | "info" | "warning" | "success" | "error"> = {
  idle: "default",
  running: "info",
  stopping: "warning",
  finished: "success",
  stopped: "warning",
  failed: "error",
};

export default function TestRunPage() {
  const files = useRunFiles();
  const start = useStartRun();
  const stop = useStopRun();
  const { lines, run, error, pollNow } = useRunLog();

  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [judgeWorkers, setJudgeWorkers] = useState(4); // pair-judging model calls made at the same time
  const initialised = useRef(false);

  const paths = useMemo(() => (files.data ?? []).map((f) => f.path), [files.data]);
  const tree = useMemo(() => buildTree(paths), [paths]);

  useEffect(() => {
    if (files.data && !initialised.current) {
      initialised.current = true;
      setSelected(new Set(files.data.filter((f) => f.selected_by_default).map((f) => f.path)));
      setExpanded(new Set(allFolders(buildTree(files.data.map((f) => f.path)))));
    }
  }, [files.data]);

  const active = run.state === "running" || run.state === "stopping";

  const toggleFiles = (toggled: string[], checked: boolean) =>
    setSelected((prev) => {
      const next = new Set(prev);
      toggled.forEach((p) => (checked ? next.add(p) : next.delete(p)));
      return next;
    });
  const toggleExpanded = (folder: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (!next.delete(folder)) next.add(folder);
      return next;
    });

  // keep the newest line in view, unless the reader has scrolled up
  const logRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);
  useEffect(() => {
    const el = logRef.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [lines]);

  const onStart = () =>
    start.mutate({ documents: paths.filter((p) => selected.has(p)), judgeWorkers }, { onSuccess: () => void pollNow() });
  const onStop = () => stop.mutate(undefined, { onSuccess: () => void pollNow() });

  return (
    <Box>
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 2, flexWrap: "wrap", gap: 1 }}>
        <Box>
          <Typography variant="h5">Test run</Typography>
          <Typography variant="body2" color="text.secondary">
            Runs the current pipeline configuration on the checked documents. The log refreshes every {POLL_MS / 1000} seconds.
          </Typography>
        </Box>
        <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
          <Chip label={run.state} color={STATE_COLOR[run.state]} />
          {active && <CircularProgress size={18} />}
          <TextField
            select
            size="small"
            id="judge-workers"
            label="Judge calls at once"
            value={judgeWorkers}
            disabled={active}
            onChange={(e) => setJudgeWorkers(Number(e.target.value))}
            sx={{ minWidth: 150 }}
          >
            {[1, 2, 4, 8].map((n) => (
              <MenuItem key={n} value={n}>{n === 1 ? "1 (one by one)" : n}</MenuItem>
            ))}
          </TextField>
          <Button
            variant="contained"
            startIcon={<PlayArrowIcon />}
            disabled={active || start.isPending || selected.size === 0}
            onClick={onStart}
          >
            Start
          </Button>
          <Button
            variant="outlined"
            color="error"
            startIcon={<StopIcon />}
            disabled={run.state !== "running" || stop.isPending}
            onClick={onStop}
          >
            Stop
          </Button>
        </Stack>
      </Box>

      {start.isError && <Alert severity="error" sx={{ mb: 2 }}>{errorMessage(start.error)}</Alert>}
      {stop.isError && <Alert severity="error" sx={{ mb: 2 }}>{errorMessage(stop.error)}</Alert>}
      {error && <Alert severity="warning" sx={{ mb: 2 }}>Could not refresh the log: {error}</Alert>}
      {run.state === "stopping" && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Stopping: the run finishes its current step, then writes its results file with what it did so far.
        </Alert>
      )}
      {(run.state === "finished" || run.state === "stopped" || run.state === "failed") && (
        <Alert severity={run.state === "failed" ? "error" : run.state === "stopped" ? "warning" : "success"} sx={{ mb: 2 }}>
          Run {run.state} (exit code {run.exit_code}). {run.results_file ? `Results file: ${run.results_file}` : "No results file was reported."}
        </Alert>
      )}

      <Box sx={{ display: "grid", gap: 2, gridTemplateColumns: { xs: "1fr", md: "minmax(280px, 1fr) 2fr" } }}>
        <Paper variant="outlined" sx={{ p: 1.5, maxHeight: 560, overflow: "auto" }}>
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1 }}>
            <Typography variant="subtitle2">
              Documents ({selected.size} of {paths.length} included)
            </Typography>
            <Stack direction="row" spacing={0.5}>
              <Button size="small" disabled={active} onClick={() => setSelected(new Set(paths))}>All</Button>
              <Button size="small" disabled={active} onClick={() => setSelected(new Set())}>None</Button>
            </Stack>
          </Box>
          {files.isLoading && <CircularProgress size={20} />}
          {files.isError && <Alert severity="error">{errorMessage(files.error)}</Alert>}
          {files.data && paths.length === 0 && (
            <Typography variant="body2" color="text.secondary">No supported documents found in the data room folder.</Typography>
          )}
          <FileTree
            nodes={tree}
            selected={selected}
            expanded={expanded}
            disabled={active}
            onToggleFiles={toggleFiles}
            onToggleExpanded={toggleExpanded}
          />
        </Paper>

        <Paper variant="outlined" sx={{ p: 0, display: "flex", flexDirection: "column", minHeight: 320 }}>
          <Typography variant="subtitle2" sx={{ px: 1.5, py: 1, borderBottom: 1, borderColor: "divider" }}>
            Live log {run.run_id ? `(${run.run_id})` : ""}
          </Typography>
          <Box
            ref={logRef}
            onScroll={(e) => {
              const el = e.currentTarget;
              stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
            }}
            sx={{ flex: 1, maxHeight: 520, overflow: "auto", p: 1.5, bgcolor: "grey.900", color: "grey.100" }}
          >
            <Box component="pre" sx={{ m: 0, fontFamily: "monospace", fontSize: 12.5, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
              {lines.length > 0
                ? lines.join("\n")
                : run.state === "idle"
                  ? "No run yet. Check the documents to include, then press Start."
                  : "Waiting for the first lines…"}
            </Box>
          </Box>
        </Paper>
      </Box>
    </Box>
  );
}
