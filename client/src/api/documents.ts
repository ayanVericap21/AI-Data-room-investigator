import { http } from "./client";
import type { DocumentList, ScanResult } from "../types/document";

export const fetchDocuments = async (): Promise<DocumentList> => (await http.get<DocumentList>("/documents")).data;

export const scanDocuments = async (): Promise<ScanResult> => (await http.post<ScanResult>("/documents/scan")).data;
