export interface PhaseMetrics {
  phase: string;
  attempts: number;
  failed: number;
  prompt_tokens: number;
  completion_tokens: number;
  thinking_tokens: number;
  avg_context_tokens: number;
  max_context_tokens: number;
  avg_ms: number;
  p95_ms: number;
  total_s: number;
}

export interface StageTiming {
  stage: string;
  seconds: number;
  percent: number;
}

export interface RunMetrics {
  investigation_id: string | null;
  config_name: string;
  state: string;
  started_at: string;
  finished_at: string;
  wall_s: number;
  results_file: string;
  settings: {
    extractor: string;
    pairing: string;
    vocabulary: string;
    coverage_check: string;
    batch_chunks: number | null;
    batch_chars: number;
    llm_model: string;
    embedding_model: string;
  };
  counts: {
    documents: number;
    documents_fully_read: number;
    chunks: number;
    claims_recorded: number;
    claims_rejected: number;
    candidate_pairs: number;
    pairs_without_clustering: number;
    extractor_credits: number;
    pairs_exact: number;
    pairs_judged: number;
    verdicts: Record<string, number>;
    findings_saved: number;
    findings_rejected: number;
    embedding_calls: number;
    embedding_texts: number;
    coverage_calls: number;
    coverage_claims_added: number;
  };
  llm: {
    attempts: number;
    failed: number;
    retries: number;
    total_s: number;
    avg_ms: number;
    p95_ms: number;
    max_ms: number;
    prompt_tokens: number;
    completion_tokens: number;
    thinking_tokens: number;
    cached_tokens: number;
    total_tokens: number;
    credits: number;
    min_context_tokens: number;
    avg_context_tokens: number;
    max_context_tokens: number;
  };
  phases: PhaseMetrics[];
  stages: StageTiming[];
}
