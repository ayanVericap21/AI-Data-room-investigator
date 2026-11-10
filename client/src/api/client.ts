import axios from "axios";

export const http = axios.create({ baseURL: "/api" });

/** Pulls the backend's consistent {error, message} shape out of an Axios error. */
export function errorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    return err.response?.data?.message ?? err.message;
  }
  return err instanceof Error ? err.message : "Unknown error";
}
