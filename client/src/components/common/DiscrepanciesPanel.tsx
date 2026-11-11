import { useState } from "react";
import {
  Accordion, AccordionDetails, AccordionSummary, Alert, Box, Button, Chip, CircularProgress, MenuItem, Paper, Select, Stack, Typography,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { errorMessage } from "../../api/client";
import { useFindings, useInvestigations, useRunMetrics } from "../../hooks/useFindings";
import type { Finding } from "../../types/finding";
import { formatDateTime } from "../../utils/format";
import { locationLabel } from "../../utils/location";
import RunMetricsPanel from "./RunMetricsPanel";

const CATEGORY_LABEL: Record<string, string> = {
  contradiction: "Contradictions",
  contextual_difference: "Contextual differences",
  missing_information: "Missing information",
  insufficient_evidence: "Insufficient evidence",
  consistent: "Consistent",
};
const CATEGORY_COLOR: Record<string, "error" | "warning" | "info" | "default" | "success"> = {
  contradiction: "error", contextual_difference: "warning", missing_information: "info", insufficient_evidence: "default", consistent: "success",
};
const SEVERITY_COLOR: Record<string, "error" | "warning" | "info" | "default"> = {
  critical: "error", high: "error", medium: "warning", low: "info", informational: "default",
};
const ROLE_COLOR: Record<string, "error" | "info" | "success" | "default"> = {
  contradicting: "error", contextual: "info", supporting: "success", insufficient: "default",
};

const categoryLabel = (c: string) => CATEGORY_LABEL[c] ?? c.replace(/_/g, " ");

function FindingCard({ finding: f, onOpenSource }: { finding: Finding; onOpenSource: (chunkId: string) => void }) {
  return (
    <Paper variant="outlined" sx={{ p: 1.5, mb: 1.5 }}>
      <Stack direction="row" spacing={1} sx={{ alignItems: "center", flexWrap: "wrap", rowGap: 0.5, mb: 0.5 }}>
        <Chip size="small" label={f.severity} color={SEVERITY_COLOR[f.severity] ?? "default"} />
        <Typography variant="subtitle2" sx={{ flex: 1, minWidth: 200 }}>{f.title}</Typography>
        <Typography variant="caption" color="text.secondary">confidence {f.confidence.toFixed(2)} · {f.verification_status}</Typography>
      </Stack>
      <Typography variant="body2" sx={{ mb: 1 }}>{f.description}</Typography>
      {f.materiality_reasoning && (
        <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>Why it matters: {f.materiality_reasoning}</Typography>
      )}
      <Box sx={{ display: "grid", gap: 1, gridTemplateColumns: { xs: "1fr", md: "repeat(auto-fit, minmax(280px, 1fr))" } }}>
        {f.evidence.map((e, i) => (
          <Box key={`${e.source.chunk_id}-${i}`} sx={{ borderLeft: 3, borderColor: "divider", pl: 1.5 }}>
            <Stack direction="row" spacing={1} sx={{ alignItems: "center", flexWrap: "wrap", rowGap: 0.5 }}>
              <Chip size="small" variant="outlined" label={e.role} color={ROLE_COLOR[e.role] ?? "default"} />
              <Typography variant="caption" sx={{ fontWeight: 600 }}>{e.source.filename}</Typography>
              <Typography variant="caption" color="text.secondary">{locationLabel(e.source)}</Typography>
            </Stack>
            <Typography variant="body2" sx={{ fontStyle: "italic", my: 0.5, wordBreak: "break-word" }}>“{e.quoted_text}”</Typography>
            <Button size="small" onClick={() => onOpenSource(e.source.chunk_id)}>View full passage</Button>
          </Box>
        ))}
      </Box>
    </Paper>
  );
}

/** The discrepancies a run found, by category, with their evidence, plus the run's metrics. */
export default function DiscrepanciesPanel({ onOpenSource }: { onOpenSource: (chunkId: string) => void }) {
  const [runId, setRunId] = useState<string>("");
  const runs = useInvestigations();
  const findings = useFindings(runId || undefined);
  const shown = findings.data?.investigation ?? null;
  const metrics = useRunMetrics(shown?.id);

  if (findings.isLoading) return <CircularProgress size={22} />;
  if (findings.isError) return <Alert severity="error">{errorMessage(findings.error)}</Alert>;
  if (!shown || !findings.data) {
    return <Alert severity="info">No run yet. Start one from the Test run page; its discrepancies will appear here.</Alert>;
  }

  const { counts, findings: items } = findings.data;
  const categories = Object.keys(counts);

  return (
    <Box>
      <Stack direction="row" spacing={2} sx={{ alignItems: "center", flexWrap: "wrap", rowGap: 1, mb: 2 }}>
        <Select size="small" value={runId || shown.id} onChange={(e) => setRunId(e.target.value)} sx={{ minWidth: 340 }}>
          {(runs.data ?? [shown]).map((r) => (
            <MenuItem key={r.id} value={r.id}>
              {r.started_at ? formatDateTime(r.started_at) : r.name} · {r.status} · {r.finding_count} findings
            </MenuItem>
          ))}
        </Select>
        {categories.map((c) => (
          <Chip key={c} label={`${categoryLabel(c)}: ${counts[c]}`} color={CATEGORY_COLOR[c] ?? "default"} />
        ))}
        {categories.length === 0 && <Chip label="No discrepancies found" color="success" />}
      </Stack>

      <Accordion variant="outlined" disableGutters defaultExpanded={false} sx={{ mb: 2 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}><Typography variant="subtitle2">Run metrics</Typography></AccordionSummary>
        <AccordionDetails>
          {metrics.isLoading && <CircularProgress size={18} />}
          {metrics.isError && <Alert severity="error">{errorMessage(metrics.error)}</Alert>}
          {metrics.data === null && (
            <Alert severity="info">No metrics were recorded for this run. Runs started after metrics were added have them.</Alert>
          )}
          {metrics.data && <RunMetricsPanel metrics={metrics.data} />}
        </AccordionDetails>
      </Accordion>

      {categories.map((c) => (
        <Accordion key={c} variant="outlined" disableGutters defaultExpanded={c === "contradiction" || categories.length === 1} sx={{ mb: 1 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle1">{categoryLabel(c)} ({counts[c]})</Typography>
          </AccordionSummary>
          <AccordionDetails>
            {items.filter((f) => f.classification === c).map((f) => (
              <FindingCard key={f.id} finding={f} onOpenSource={onOpenSource} />
            ))}
          </AccordionDetails>
        </Accordion>
      ))}
    </Box>
  );
}
