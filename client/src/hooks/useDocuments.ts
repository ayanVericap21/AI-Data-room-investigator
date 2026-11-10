import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { fetchDocuments, scanDocuments } from "../api/documents";

export const DOCUMENTS_KEY = ["documents"] as const;

export function useDocuments() {
  return useQuery({ queryKey: DOCUMENTS_KEY, queryFn: fetchDocuments });
}

export function useScanDocuments() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: scanDocuments,
    onSuccess: () => qc.invalidateQueries({ queryKey: DOCUMENTS_KEY }),
  });
}
