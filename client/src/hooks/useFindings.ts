import { useQuery } from "@tanstack/react-query";
import { fetchFindings, fetchInvestigations, fetchRunMetrics } from "../api/findings";

const REFRESH_MS = 10000; // a run may finish while the tab is open

/** Findings of one run; without an id, the newest run. */
export const useFindings = (investigationId?: string) =>
  useQuery({ queryKey: ["findings", investigationId ?? "latest"], queryFn: () => fetchFindings(investigationId), refetchInterval: REFRESH_MS });

export const useInvestigations = () =>
  useQuery({ queryKey: ["investigations"], queryFn: fetchInvestigations, refetchInterval: REFRESH_MS });

export const useRunMetrics = (investigationId: string | undefined) =>
  useQuery({ queryKey: ["run-metrics", investigationId], queryFn: () => fetchRunMetrics(investigationId!), enabled: !!investigationId });
