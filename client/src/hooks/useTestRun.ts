import { useMutation, useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";
import { fetchRunFiles, fetchUpdates, startRun, stopRun } from "../api/testRun";
import type { RunStatus } from "../types/testRun";

export const POLL_MS = 5000; // how often the log is asked for while the page is open
const MAX_LINES = 5000; // the browser keeps at most this many lines

const IDLE: RunStatus = {
  run_id: null, state: "idle", documents: [], started_at: null, finished_at: null, exit_code: null, results_file: null,
};

export function useRunFiles() {
  return useQuery({ queryKey: ["test-run-files"], queryFn: fetchRunFiles });
}

export const useStartRun = () => useMutation({ mutationFn: startRun });
export const useStopRun = () => useMutation({ mutationFn: stopRun });

/** Polls /api/updates and appends the new log lines. Returns the lines, the run status and a "poll now" trigger. */
export function useRunLog() {
  const [lines, setLines] = useState<string[]>([]);
  const [run, setRun] = useState<RunStatus>(IDLE);
  const [error, setError] = useState<string | null>(null);
  const offset = useRef(0);
  const runId = useRef<string | null>(null);
  const busy = useRef(false);

  const poll = useCallback(async () => {
    if (busy.current) return;
    busy.current = true;
    try {
      let update = await fetchUpdates(offset.current);
      if (update.run.run_id !== runId.current) {
        // a different run: the offset belonged to the old log, so start this one from the top
        runId.current = update.run.run_id;
        offset.current = 0;
        setLines([]);
        update = await fetchUpdates(0);
      }
      offset.current = update.next_offset;
      setRun(update.run);
      if (update.lines.length > 0) setLines((prev) => [...prev, ...update.lines].slice(-MAX_LINES));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not reach the server");
    } finally {
      busy.current = false;
    }
  }, []);

  useEffect(() => {
    void poll();
    const timer = setInterval(() => void poll(), POLL_MS);
    return () => clearInterval(timer);
  }, [poll]);

  return { lines, run, error, pollNow: poll };
}
