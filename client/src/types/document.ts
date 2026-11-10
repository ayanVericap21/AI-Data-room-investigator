export interface DocumentItem {
  id: string;
  filename: string;
  file_type: "pdf" | "docx" | "xlsx";
  file_size: number;
  processing_status: string;
  last_modified_at: string;
  unit_count: number | null;
}

export interface DocumentList {
  items: DocumentItem[];
  total: number;
}

export interface ScanResult {
  discovered: number;
  added: number;
  updated: number;
  unchanged: number;
  missing: number;
  skipped_duplicates: string[];
  unsupported: string[];
  documents: DocumentItem[];
}
