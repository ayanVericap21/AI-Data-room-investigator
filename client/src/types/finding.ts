import type { SourceRef } from "./claim";

export interface Evidence {
  role: string;
  quoted_text: string;
  relevance_reason: string;
  support_assessment: string;
  source: SourceRef;
}

export interface Finding {
  id: string;
  title: string;
  description: string;
  classification: string;
  severity: string;
  confidence: number;
  evidence_status: string;
  verification_status: string;
  materiality_reasoning: string | null;
  business_impact: string | null;
  unresolved_questions: string[];
  evidence: Evidence[];
}

export interface InvestigationSummary {
  id: string;
  name: string;
  status: string;
  started_at: string | null;
  completed_at: string | null;
  finding_count: number;
}

export interface FindingsResult {
  investigation: InvestigationSummary | null;
  counts: Record<string, number>;
  findings: Finding[];
}
