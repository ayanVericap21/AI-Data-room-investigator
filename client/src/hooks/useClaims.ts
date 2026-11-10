import { useEffect, useRef } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { fetchCandidates, fetchChunk, fetchClaims, fetchJobs, startExtraction, startProcessing } from "../api/claims";
import { DOCUMENTS_KEY } from "./useDocuments";

export const CLAIMS_KEY = ["claims"] as const;
export const CANDIDATES_KEY = ["candidates"] as const;
export const JOBS_KEY = ["jobs"] as const;
const JOB_POLL_MS = 3000;

export const useClaims = () => useQuery({ queryKey: CLAIMS_KEY, queryFn: () => fetchClaims() });

export const useCandidates = () => useQuery({ queryKey: CANDIDATES_KEY, queryFn: fetchCandidates });

export const useChunk = (id: string | null) =>
  useQuery({ queryKey: ["chunk", id], queryFn: () => fetchChunk(id!), enabled: id !== null });

/** Polls only while a background job is running; refreshes data when one finishes. */
export function useJobs() {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: JOBS_KEY,
    queryFn: fetchJobs,
    refetchInterval: (q) => (q.state.data?.some((j) => j.state === "running") ? JOB_POLL_MS : false),
  });
  const wasRunning = useRef(false);
  const running = query.data?.some((j) => j.state === "running") ?? false;
  useEffect(() => {
    if (wasRunning.current && !running) {
      qc.invalidateQueries({ queryKey: DOCUMENTS_KEY });
      qc.invalidateQueries({ queryKey: CLAIMS_KEY });
      qc.invalidateQueries({ queryKey: CANDIDATES_KEY });
    }
    wasRunning.current = running;
  }, [running, qc]);
  return { ...query, running };
}

function useStartJob(fn: () => Promise<unknown>) {
  const qc = useQueryClient();
  return useMutation({ mutationFn: fn, onSuccess: () => qc.invalidateQueries({ queryKey: JOBS_KEY }) });
}

export const useStartProcessing = () => useStartJob(startProcessing);
export const useStartExtraction = () => useStartJob(startExtraction);
