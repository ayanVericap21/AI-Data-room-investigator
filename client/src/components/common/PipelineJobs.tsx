import { Alert, Box, Button, Chip, CircularProgress, Paper, Stack, Typography } from "@mui/material";
import { errorMessage } from "../../api/client";
import { useJobs, useStartExtraction, useStartProcessing } from "../../hooks/useClaims";
import type { Job } from "../../types/claim";

const LABELS: Record<string, string> = {
  process_documents: "Process documents (extract, chunk, embed)",
  extract_claims: "Extract claims",
};
const COLORS = { idle: "default", running: "info", completed: "success", failed: "error" } as const;

function JobRow({ job }: { job: Job }) {
  const rejected = (job.result?.claims_rejected as number | undefined) ?? 0;
  const notes = (job.result?.notes as string[] | undefined) ?? [];
  return (
    <Box>
      <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
        <Typography variant="body2" sx={{ minWidth: 280 }}>{LABELS[job.name] ?? job.name}</Typography>
        <Chip size="small" label={job.state} color={COLORS[job.state]} />
        {job.state === "running" && <CircularProgress size={14} />}
        <Typography variant="caption" color="text.secondary">{job.message}</Typography>
      </Stack>
      {job.state === "completed" && rejected > 0 && (
        <Typography variant="caption" color="warning.main">
          {rejected} claim(s) rejected by validation (unknown source or value not found in the cited text).
        </Typography>
      )}
      {notes.map((n) => (
        <Typography key={n} variant="caption" color="text.secondary" sx={{ display: "block" }}>{n}</Typography>
      ))}
      {job.state === "failed" && <Alert severity="error" sx={{ mt: 0.5 }}>{job.error}</Alert>}
    </Box>
  );
}

export default function PipelineJobs() {
  const jobs = useJobs();
  const process = useStartProcessing();
  const extract = useStartExtraction();
  const startError = process.error ?? extract.error;

  return (
    <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
      <Stack direction="row" spacing={1} sx={{ mb: 1.5 }}>
        <Button variant="outlined" disabled={jobs.running || process.isPending} onClick={() => process.mutate()}>
          Process documents
        </Button>
        <Button variant="contained" disabled={jobs.running || extract.isPending} onClick={() => extract.mutate()}>
          Extract claims
        </Button>
      </Stack>
      {startError && <Alert severity="error" sx={{ mb: 1 }}>{errorMessage(startError)}</Alert>}
      <Stack spacing={1}>{jobs.data?.map((j) => <JobRow key={j.name} job={j} />)}</Stack>
    </Paper>
  );
}
