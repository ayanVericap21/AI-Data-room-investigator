import { useState } from "react";
import {
  Alert, Box, Button, Chip, Dialog, DialogContent, DialogTitle, Divider, Paper, Tab, Table, TableBody, TableCell,
  Link, Stack, TableContainer, TableHead, TableRow, Tabs, Tooltip, Typography,
} from "@mui/material";
import { errorMessage } from "../../api/client";
import DiscrepanciesPanel from "./DiscrepanciesPanel";
import { useCandidates, useChunk, useClaims } from "../../hooks/useClaims";
import { useVirtualRows } from "../../hooks/useVirtualRows";
import type { Claim, RelationHint } from "../../types/claim";
import { locationLabel } from "../../utils/location";

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

export default function ClaimsPanel() {
  const [tab, setTab] = useState(0);
  const [chunkId, setChunkId] = useState<string | null>(null);
  const claims = useClaims();
  const candidates = useCandidates();
  const claimRows = useVirtualRows(tab === 0 ? claims.data?.items.length ?? 0 : 0, 76, (i) => claims.data?.items[i]?.id ?? i);
  const groupRows = useVirtualRows(tab === 1 ? candidates.data?.items.length ?? 0 : 0, 320);

  return (
    <Box>
      <Typography variant="h6" sx={{ mb: 1.5 }}>Claims and findings</Typography>
      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 1 }}>
        <Tab label={`Claims (${claims.data?.total ?? 0})`} />
        <Tab label={`Claim groups (${candidates.data?.total ?? 0})`} />
        <Tab label="Main discrepancies" />
      </Tabs>

      {tab === 0 && (
        <TableContainer component={Paper} variant="outlined" ref={claimRows.ref}>
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
              {claimRows.before > 0 && <TableRow sx={{ height: claimRows.before }}><TableCell colSpan={6} sx={{ p: 0, border: 0 }} /></TableRow>}
              {claimRows.items.map((row) => {
                const c = claims.data!.items[row.index];
                return (
                <TableRow key={c.id} hover ref={claimRows.measure} data-index={row.index}>
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
                );
              })}
              {claimRows.after > 0 && <TableRow sx={{ height: claimRows.after }}><TableCell colSpan={6} sx={{ p: 0, border: 0 }} /></TableRow>}
              {claims.data?.items.length === 0 && (
                <TableRow><TableCell colSpan={6} align="center">
                  No claims yet. Check the documents above and press Start.
                </TableCell></TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {tab === 1 && (
        <Box>
          {candidates.isError && <Alert severity="error">{errorMessage(candidates.error)}</Alert>}
          <Alert severity="info" icon={false} sx={{ mb: 1.5 }}>
            Each group is a fact and the claims that may state it, shown with the sentence each one comes from. Under every claim are its candidates: the other claims of
            its group it may be compared with, each with its source. These are candidates, not findings.
          </Alert>
          <Box ref={groupRows.ref} sx={{ pt: `${groupRows.before}px`, pb: `${groupRows.after}px` }}>
          {groupRows.items.map((row) => {
            const g = candidates.data!.items[row.index];
            const gi = row.index;
            const first = g.claims[0]?.claim;
            return (
              <Box key={gi} ref={groupRows.measure} data-index={row.index} sx={{ pb: 4 }}>
              <Paper variant="outlined" sx={{ overflow: "hidden", borderColor: "text.disabled" }}>
                <Box sx={{ px: 2, py: 1.25, bgcolor: "action.selected", borderBottom: 1, borderColor: "text.disabled", display: "flex", alignItems: "center", gap: 1 }}>
                  <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
                    Group {gi + 1}: {first?.metric ?? first?.subject ?? "claims"}
                  </Typography>
                  <Chip size="small" label={`${g.claims.length} claims`} />
                </Box>
                <Stack divider={<Divider flexItem />}>
                  {g.claims.map((gc) => {
                    const c = gc.claim;
                    return (
                      <Box key={gc.claim.id} sx={{ px: 2, py: 1.5 }}>
                        <Stack direction="row" spacing={1} sx={{ alignItems: "center", flexWrap: "wrap" }}>
                          {c ? (
                            <Link component="button" type="button" underline="hover" variant="body2" sx={{ fontWeight: 600, textAlign: "left" }}
                              onClick={() => setChunkId(c.source.chunk_id)}>
                              {c.source.filename} · {locationLabel(c.source)}
                            </Link>
                          ) : <Typography variant="body2">unknown claim</Typography>}
                        </Stack>
                        <Box sx={{ borderLeft: 3, borderColor: "divider", pl: 1.5, ml: 0.5, mt: 0.75 }}>
                          <Typography variant="body2" color="text.secondary">{c?.claim_text || describe(c)}</Typography>
                          {c && (
                            <Typography variant="caption" color="text.disabled" sx={{ display: "block", mt: 0.25 }}>
                              {c.subject ?? "?"} · {c.metric ?? "?"}: {c.value ?? "?"} {c.unit ?? ""} {c.period ? `(${c.period})` : ""}
                            </Typography>
                          )}
                        </Box>
                        {gc.candidates.length > 0 && (
                          <Box sx={{ mt: 1, ml: 0.5 }}>
                            <Typography variant="caption" color="text.secondary">Candidates</Typography>
                            <Stack spacing={1} sx={{ mt: 0.5 }}>
                              {gc.candidates.map((k) => {
                                const other = k.claim;
                                return (
                                  <Box key={k.claim.id} sx={{ pl: 1.5, borderLeft: 3, borderColor: "divider" }}>
                                    <Stack direction="row" spacing={1} sx={{ alignItems: "center", flexWrap: "wrap", rowGap: 0.5 }}>
                                      {other ? (
                                        <Link component="button" type="button" underline="hover" variant="body2" sx={{ fontWeight: 600, textAlign: "left" }}
                                          onClick={() => setChunkId(other.source.chunk_id)}>
                                          {other.source.filename} · {locationLabel(other.source)}
                                        </Link>
                                      ) : <Typography variant="body2">unknown claim</Typography>}
                                      <Tooltip title={`found by ${k.matched_by} · score ${k.topic_score.toFixed(2)}`}>
                                        <Chip size="small" variant="outlined" color={HINT_COLOR[k.relation_hint]} label={k.relation_hint} />
                                      </Tooltip>
                                    </Stack>
                                    <Typography variant="body2" color="text.secondary" sx={{ mt: 0.25 }}>{other?.claim_text || describe(other)}</Typography>
                                  </Box>
                                );
                              })}
                            </Stack>
                          </Box>
                        )}
                      </Box>
                    );
                  })}
                </Stack>
              </Paper>
              </Box>
            );
          })}
          </Box>
          {candidates.data?.items.length === 0 && (
            <Paper variant="outlined" sx={{ p: 2, textAlign: "center" }}>No claim groups.</Paper>
          )}
        </Box>
      )}
      {tab === 2 && <DiscrepanciesPanel onOpenSource={setChunkId} />}
      <SourceDialog chunkId={chunkId} onClose={() => setChunkId(null)} />
    </Box>
  );
}
