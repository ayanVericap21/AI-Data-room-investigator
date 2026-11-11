import { Box, Chip, Paper, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography } from "@mui/material";
import type { RunMetrics } from "../../types/runMetrics";

const n = (v?: number) => (v ?? 0).toLocaleString(); // a field missing from an older response shows 0, not a crash

function duration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const m = Math.floor(seconds / 60);
  return `${m} min ${Math.round(seconds - m * 60)} s`;
}

function Stat({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <Paper variant="outlined" sx={{ p: 1.5 }}>
      <Typography variant="caption" color="text.secondary">{label}</Typography>
      <Typography variant="h6" sx={{ lineHeight: 1.2 }}>{value}</Typography>
      {detail && <Typography variant="caption" color="text.secondary">{detail}</Typography>}
    </Paper>
  );
}

/** The same numbers as the run's results file: time, calls, tokens, latencies, counts. */
export default function RunMetricsPanel({ metrics: m }: { metrics: RunMetrics }) {
  const verdicts = Object.entries(m.counts.verdicts).map(([k, v]) => `${k.replace(/_/g, " ")}: ${v}`).join(", ");
  return (
    <Box>
      <Stack direction="row" spacing={1} sx={{ mb: 1.5, flexWrap: "wrap", rowGap: 1 }}>
        <Chip size="small" label={m.state} color={m.state === "completed" ? "success" : m.state === "stopped" ? "warning" : "error"} />
        <Chip size="small" variant="outlined" label={`extractor: ${m.settings.extractor}`} />
        <Chip size="small" variant="outlined" label={`pairing: ${m.settings.pairing}`} />
        <Chip size="small" variant="outlined" label={`vocabulary: ${m.settings.vocabulary}`} />
        <Chip size="small" variant="outlined" label={`coverage check: ${m.settings.coverage_check}`} />
        <Chip size="small" variant="outlined" label={`chunks per call: ${m.settings.batch_chunks ?? "by size"}`} />
        <Chip size="small" variant="outlined" label={m.settings.llm_model} />
      </Stack>

      <Box sx={{ display: "grid", gap: 1.5, gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", mb: 2 }}>
        <Stat label="Total time" value={duration(m.wall_s)} detail={`LLM time ${duration(m.llm.total_s)}`} />
        <Stat label="Documents" value={`${m.counts.documents_fully_read} / ${m.counts.documents}`} detail={`fully read · ${n(m.counts.chunks)} chunks`} />
        <Stat label="Claims" value={n(m.counts.claims_recorded)} detail={`${n(m.counts.claims_rejected)} rejected`} />
        <Stat label="Candidate pairs" value={n(m.counts.candidate_pairs)} detail={`${n(m.counts.pairs_exact)} exact · ${n(m.counts.pairs_judged)} judged`
          + (m.counts.pairs_without_clustering > 0 ? ` · ${n(m.counts.pairs_without_clustering)} before grouping by value` : "")} />
        <Stat label="Findings" value={n(m.counts.findings_saved)} detail={`${n(m.counts.findings_rejected)} rejected (quotes not verified)`} />
        <Stat label="LLM calls" value={n(m.llm.attempts)} detail={`${m.llm.failed} failed · ${m.llm.retries} retries`} />
        <Stat label="Latency per call" value={`${n(m.llm.avg_ms)} ms avg`} detail={`p95 ${n(m.llm.p95_ms)} ms · max ${n(m.llm.max_ms)} ms`} />
        <Stat label="Tokens" value={n(m.llm.total_tokens)} detail={`prompt ${n(m.llm.prompt_tokens)} · completion ${n(m.llm.completion_tokens)} · cached ${n(m.llm.cached_tokens)}`} />
        <Stat label="Context per call" value={`${n(m.llm.avg_context_tokens)} tokens avg`} detail={`min ${n(m.llm.min_context_tokens)} · max ${n(m.llm.max_context_tokens)}`} />
        <Stat label="Embeddings" value={n(m.counts.embedding_calls)} detail={`${n(m.counts.embedding_texts)} texts (local)`} />
        {(m.settings.extractor === "llamaextract" || m.llm.credits > 0) && (
          <Stat label="LlamaExtract credits" value={n(m.counts.extractor_credits + m.llm.credits)}
            detail={`extraction ${n(m.counts.extractor_credits)} · grouping and judging ${n(m.llm.credits)}`} />
        )}
        {m.settings.coverage_check === "on" && (
          <Stat label="Coverage follow-ups" value={n(m.counts.coverage_calls)} detail={`${n(m.counts.coverage_claims_added)} claims added`} />
        )}
      </Box>
      {verdicts && <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>Judge verdicts — {verdicts}</Typography>}

      <Typography variant="subtitle2" sx={{ mb: 0.5 }}>LLM calls by phase</Typography>
      <TableContainer component={Paper} variant="outlined" sx={{ mb: 2 }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Phase</TableCell><TableCell align="right">Calls</TableCell><TableCell align="right">Failed</TableCell>
              <TableCell align="right">Prompt tok</TableCell><TableCell align="right">Completion tok</TableCell>
              <TableCell align="right">Avg ctx tok</TableCell><TableCell align="right">Max ctx tok</TableCell>
              <TableCell align="right">Avg ms</TableCell><TableCell align="right">p95 ms</TableCell><TableCell align="right">Total s</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {m.phases.map((p) => (
              <TableRow key={p.phase}>
                <TableCell>{p.phase}</TableCell><TableCell align="right">{p.attempts}</TableCell><TableCell align="right">{p.failed}</TableCell>
                <TableCell align="right">{n(p.prompt_tokens)}</TableCell><TableCell align="right">{n(p.completion_tokens)}</TableCell>
                <TableCell align="right">{n(p.avg_context_tokens)}</TableCell><TableCell align="right">{n(p.max_context_tokens)}</TableCell>
                <TableCell align="right">{n(p.avg_ms)}</TableCell><TableCell align="right">{n(p.p95_ms)}</TableCell><TableCell align="right">{p.total_s}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <Typography variant="subtitle2" sx={{ mb: 0.5 }}>Time by stage</Typography>
      <TableContainer component={Paper} variant="outlined" sx={{ mb: 1 }}>
        <Table size="small">
          <TableHead>
            <TableRow><TableCell>Stage</TableCell><TableCell align="right">Seconds</TableCell><TableCell align="right">% of total</TableCell></TableRow>
          </TableHead>
          <TableBody>
            {m.stages.map((s) => (
              <TableRow key={s.stage}>
                <TableCell>{s.stage}</TableCell><TableCell align="right">{s.seconds.toFixed(1)}</TableCell><TableCell align="right">{s.percent.toFixed(1)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      <Typography variant="caption" color="text.secondary">Full report: {m.results_file}</Typography>
    </Box>
  );
}
