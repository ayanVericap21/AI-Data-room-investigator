export interface DataRoomFile {
  path: string;
  selected_by_default: boolean;
}

export type RunState = "idle" | "running" | "stopping" | "finished" | "stopped" | "failed";

export interface RunStatus {
  run_id: string | null;
  state: RunState;
  documents: string[];
  started_at: string | null;
  finished_at: string | null;
  exit_code: number | null;
  results_file: string | null;
}

export interface Updates {
  run: RunStatus;
  lines: string[];
  next_offset: number;
}
