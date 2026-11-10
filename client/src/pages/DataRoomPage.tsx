import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import { errorMessage } from "../api/client";
import { useDocuments, useScanDocuments } from "../hooks/useDocuments";
import { formatBytes, formatDateTime } from "../utils/format";

const STATUS_COLOR: Record<string, "default" | "success" | "error" | "warning" | "info"> = {
  pending: "default",
  processing: "info",
  processed: "success",
  failed: "error",
};

export default function DataRoomPage() {
  const docs = useDocuments();
  const scan = useScanDocuments();

  return (
    <Box>
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 2 }}>
        <Box>
          <Typography variant="h5">Data room</Typography>
          <Typography variant="body2" color="text.secondary">
            {docs.data ? `${docs.data.total} document(s) available` : "Loading documents…"}
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={scan.isPending ? <CircularProgress size={16} color="inherit" /> : <RefreshIcon />}
          disabled={scan.isPending}
          onClick={() => scan.mutate()}
        >
          Scan / refresh documents
        </Button>
      </Box>

      {scan.isError && <Alert severity="error" sx={{ mb: 2 }}>{errorMessage(scan.error)}</Alert>}
      {scan.data && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Scan finished: {scan.data.added} added, {scan.data.updated} updated, {scan.data.unchanged} unchanged,{" "}
          {scan.data.missing} missing.
          {scan.data.skipped_duplicates.length > 0 &&
            ` Skipped duplicate content: ${scan.data.skipped_duplicates.join(", ")}.`}
          {scan.data.unsupported.length > 0 && ` Unsupported files ignored: ${scan.data.unsupported.join(", ")}.`}
        </Alert>
      )}
      {docs.isError && <Alert severity="error">{errorMessage(docs.error)}</Alert>}

      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>File</TableCell>
              <TableCell>Type</TableCell>
              <TableCell align="right">Size</TableCell>
              <TableCell align="right">Pages / sheets</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Last modified</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {docs.data?.items.map((d) => (
              <TableRow key={d.id} hover>
                <TableCell>{d.filename}</TableCell>
                <TableCell>{d.file_type.toUpperCase()}</TableCell>
                <TableCell align="right">{formatBytes(d.file_size)}</TableCell>
                <TableCell align="right">{d.unit_count ?? "—"}</TableCell>
                <TableCell>
                  <Chip size="small" label={d.processing_status} color={STATUS_COLOR[d.processing_status] ?? "default"} />
                </TableCell>
                <TableCell>{formatDateTime(d.last_modified_at)}</TableCell>
              </TableRow>
            ))}
            {docs.data?.items.length === 0 && (
              <TableRow>
                <TableCell colSpan={6} align="center">
                  No documents found. Put PDF, DOCX or XLSX files in server/data_room/ and click Scan.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}
