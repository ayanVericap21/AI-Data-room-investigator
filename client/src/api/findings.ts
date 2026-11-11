import axios from "axios";
import { http } from "./client";
import type { FindingsResult, InvestigationSummary } from "../types/finding";
import type { RunMetrics } from "../types/runMetrics";

export const fetchFindings = async (investigationId?: string): Promise<FindingsResult> =>
  (await http.get<FindingsResult>("/findings", { params: investigationId ? { investigation_id: investigationId } : {} })).data;

export const fetchInvestigations = async (): Promise<InvestigationSummary[]> =>
  (await http.get<InvestigationSummary[]>("/investigations")).data;

/** Metrics of a run, or null when none were recorded for it (runs from before they existed). */
export async function fetchRunMetrics(investigationId: string): Promise<RunMetrics | null> {
  try {
    return (await http.get<RunMetrics>(`/investigations/${investigationId}/metrics`)).data;
  } catch (err) {
    if (axios.isAxiosError(err) && err.response?.status === 404) return null;
    throw err;
  }
}
