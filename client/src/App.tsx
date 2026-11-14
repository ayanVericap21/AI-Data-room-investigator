import { AppBar, Container, Toolbar, Typography } from "@mui/material";
import { Navigate, Route, Routes } from "react-router-dom";
import TestRunPage from "./pages/TestRunPage";

export default function App() {
  return (
    <>
      <AppBar position="static" color="default" elevation={0} sx={{ borderBottom: 1, borderColor: "divider" }}>
        <Toolbar>
          <Typography variant="h6">Data Room Investigator</Typography>
        </Toolbar>
      </AppBar>
      <Container maxWidth="lg" sx={{ py: 3 }}>
        <Routes>
          <Route path="/" element={<TestRunPage />} />
          <Route path="/test-run" element={<TestRunPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Container>
    </>
  );
}
