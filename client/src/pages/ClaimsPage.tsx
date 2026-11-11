import { useState } from "react";
import {
  Alert, Box, Button, Chip, Dialog, DialogContent, DialogTitle, Paper, Tab, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Tabs, Typography,
} from "@mui/material";
import { errorMessage } from "../api/client";
import DiscrepanciesPanel from "../components/common/DiscrepanciesPanel";
import PipelineJobs from "../components/common/PipelineJobs";
import { useCandidates, useChunk, useClaims } from "../hooks/useClaims";
import type { Claim, RelationHint } from "../types/claim";
import { locationLabel } from "../utils/location";

const HINT_COLOR: Record<RelationHint, "default" | "success" | "warning" | "info"> = {
  exact: "success", conflicting: "warning", contextual: "info", partial: "default",
};

function SourceDialog({ chunkId, onClose }: { chunkId: string | null; onClose: () => void }) {
  const chunk = useChunk(chunkId);
  return (
    <Dialog open={chunkId !== null} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        {chunk.data ? `${chunk.data.filename} — ${locationLabel({ ...chunk.data, chunk_id: chunk.data.id })}` : "Source passage"}
      </DialogTitle>
      <DialogContent>
        {chunk.isError && <Alert severity="error">{errorMessage(chunk.error)}</Alert>}
        <Typography component="pre" variant="body2" sx={{ whiteSpace: "pre-wrap", fontFamily: "inherit" }}>
          {chunk.data?.content}
        </Typography>
      </DialogContent>
    </Dialog>
  );
}

const describe = (c: Claim | undefined) =>
  c ? `${c.subject ?? "?"} · ${c.metric ?? "?"}: ${c.value ?? "?"} ${c.unit ?? ""} ${c.period ? `(${c.period})` : ""}`.trim() : "unknown claim";

export default function ClaimsPage() {
  const [tab, setTab] = useState(0);
  const [chunkId, setChunkId] = useState<string | null>(null);
  const claims = useClaims();
  const candidates = useCandidates();
  const byId = new Map(claims.data?.items.map((c) => [c.id, c]));

  return (
    <Box>
      <Typography variant="h5" sx={{ mb: 2 }}>Claims</Typography>
      <PipelineJobs />
      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 1 }}>
        <Tab label={`Claims (${claims.data?.total ?? 0})`} />
        <Tab label={`Candidate pairs (${candidates.data?.total ?? 0})`} />
        <Tab label="Main discrepancies" />
      </Tabs>

      {tab === 0 && (
        <TableContainer component={Paper} variant="outlined">
          {claims.isError && <Alert severity="error">{errorMessage(claims.error)}</Alert>}
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Subject / metric</TableCell><TableCell>Value</TableCell><TableCell>Period</TableCell>
                <TableCell>Conditions / assumptions</TableCell><TableCell>Source</TableCell>
                <TableCell align="right">Conf.</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {claims.data?.items.map((c) => (
                <TableRow key={c.id} hover>
                  <TableCell>
                    {c.subject ?? "—"}<br /><Typography variant="caption" color="text.secondary">{c.metric ?? "—"}</Typography>
                  </TableCell>
                  <TableCell>{c.value ?? "—"} {c.unit ?? ""}</TableCell>
                  <TableCell>{c.period ?? "—"}</TableCell>
                  <TableCell>
                    {c.conditions ?? "—"}
                    {c.assumptions?.map((a) => <Chip key={a} size="small" label={a} sx={{ ml: 0.5 }} />)}
                  </TableCell>
                  <TableCell>
                    <Button size="small" onClick={() => setChunkId(c.source.chunk_id)}>
                      {c.source.filename} · {locationLabel(c.source)}
                    </Button>
                  </TableCell>
                  <TableCell align="right">{c.extraction_confidence?.toFixed(2) ?? "—"}</TableCell>
                </TableRow>
              ))}
              {claims.data?.items.length === 0 && (
                <TableRow><TableCell colSpan={6} align="center">
                  No claims yet. Scan documents, then Process documents, then Extract claims.
                </TableCell></TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {tab === 1 && (
        <TableContainer component={Paper} variant="outlined">
          {candidates.isError && <Alert severity="error">{errorMessage(candidates.error)}</Alert>}
          <Alert severity="info" icon={false}>
            Candidates are shortlisted comparisons, not findings. The hint only describes how the attributes compare.
          </Alert>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Claim A</TableCell><TableCell>Claim B</TableCell><TableCell>Hint</TableCell>
                <TableCell>Matched by</TableCell><TableCell align="right">Score</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {candidates.data?.items.map((p) => (
                <TableRow key={`${p.claim_a_id}-${p.claim_b_id}`} hover>
                  <TableCell>{describe(byId.get(p.claim_a_id))}<br />
                    <Typography variant="caption" color="text.secondary">{byId.get(p.claim_a_id)?.source.filename}</Typography></TableCell>
                  <TableCell>{describe(byId.get(p.claim_b_id))}<br />
                    <Typography variant="caption" color="text.secondary">{byId.get(p.claim_b_id)?.source.filename}</Typography></TableCell>
                  <TableCell><Chip size="small" label={p.relation_hint} color={HINT_COLOR[p.relation_hint]} /></TableCell>
                  <TableCell>{p.matched_by}</TableCell>
                  <TableCell align="right">{p.topic_score.toFixed(2)}</TableCell>
                </TableRow>
              ))}
              {candidates.data?.items.length === 0 && (
                <TableRow><TableCell colSpan={5} align="center">No candidate pairs.</TableCell></TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      )}
      {tab === 2 && <DiscrepanciesPanel onOpenSource={setChunkId} />}
      <SourceDialog chunkId={chunkId} onClose={() => setChunkId(null)} />
    </Box>
  );
}
