import { AppBar, Button, Container, Toolbar, Typography } from "@mui/material";
import { Link as RouterLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import ClaimsPage from "./pages/ClaimsPage";
import DataRoomPage from "./pages/DataRoomPage";
import TestRunPage from "./pages/TestRunPage";

const NAV = [
  { to: "/", label: "Data room" },
  { to: "/claims", label: "Claims" },
  { to: "/test-run", label: "Test run" },
];

export default function App() {
  const { pathname } = useLocation();
  return (
    <>
      <AppBar position="static" color="default" elevation={0} sx={{ borderBottom: 1, borderColor: "divider" }}>
        <Toolbar>
          <Typography variant="h6" sx={{ mr: 4 }}>Data Room Investigator</Typography>
          {NAV.map((n) => (
            <Button key={n.to} component={RouterLink} to={n.to} color={pathname === n.to ? "primary" : "inherit"}>
              {n.label}
            </Button>
          ))}
        </Toolbar>
      </AppBar>
      <Container maxWidth="lg" sx={{ py: 3 }}>
        <Routes>
          <Route path="/" element={<DataRoomPage />} />
          <Route path="/claims" element={<ClaimsPage />} />
          <Route path="/test-run" element={<TestRunPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Container>
    </>
  );
}
