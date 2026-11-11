import { http } from "./client";
import type { DataRoomFile, RunStatus, Updates } from "../types/testRun";

export const fetchRunFiles = async (): Promise<DataRoomFile[]> =>
  (await http.get<{ files: DataRoomFile[] }>("/test-run/files")).data.files;

export const startRun = async ({ documents, judgeWorkers }: { documents: string[]; judgeWorkers: number }): Promise<RunStatus> =>
  (await http.post<RunStatus>("/test-run/start", { documents, judge_workers: judgeWorkers })).data;

export const stopRun = async (): Promise<RunStatus> => (await http.post<RunStatus>("/test-run/stop")).data;

export const fetchUpdates = async (offset: number): Promise<Updates> =>
  (await http.get<Updates>("/updates", { params: { offset } })).data;
