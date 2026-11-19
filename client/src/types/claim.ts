export interface SourceRef {
  document_id: string;
  filename: string;
  chunk_id: string;
  content_type: string | null;
  page_number: number | null;
  sheet_name: string | null;
  cell_reference: string | null;
  heading: string | null;
}

export interface Claim {
  id: string;
  subject: string | null;
  metric: string | null;
  value: number | string | null;
  unit: string | null;
  period: string | null;
  conditions: string | null;
  assumptions: string[] | null;
  claim_text: string;
  claim_category: string | null;
  extraction_confidence: number | null;
  source: SourceRef;
}

export interface ClaimList {
  items: Claim[];
  total: number;
}

export interface Chunk {
  id: string;
  document_id: string;
  filename: string;
  content: string;
  content_type: string;
  page_number: number | null;
  sheet_name: string | null;
  cell_reference: string | null;
  heading: string | null;
}

/** A fact and the claims that may state it, with the route that found them. */
export interface ClaimGroup {
  claims: Claim[];
  topic: string;
  matched_by: "lexical" | "label" | "embedding" | "llm" | "identifier" | "rule";
}

export interface ClaimGroupList {
  items: ClaimGroup[];
  total: number; // groups
}

export interface Job {
  name: string;
  state: "idle" | "running" | "completed" | "failed";
  message: string;
  started_at: string | null;
  finished_at: string | null;
  result: Record<string, unknown> | null;
  error: string | null;
}
