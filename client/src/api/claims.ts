import { http } from "./client";
import type { ClaimGroupList, Chunk, ClaimList, Job } from "../types/claim";

export const fetchClaims = async (limit = 200, offset = 0): Promise<ClaimList> =>
  (await http.get<ClaimList>("/claims", { params: { limit, offset } })).data;

export const fetchCandidates = async (): Promise<ClaimGroupList> =>
  (await http.get<ClaimGroupList>("/claims/candidates")).data;

export const fetchChunk = async (id: string): Promise<Chunk> => (await http.get<Chunk>(`/chunks/${id}`)).data;

export const fetchJobs = async (): Promise<Job[]> => (await http.get<Job[]>("/jobs")).data;

export const startProcessing = async (): Promise<Job> => (await http.post<Job>("/documents/process")).data;

export const clearResults = async (): Promise<Record<string, number>> => (await http.delete<Record<string, number>>("/results")).data;

export const startExtraction = async (): Promise<Job> => (await http.post<Job>("/claims/extract")).data;
