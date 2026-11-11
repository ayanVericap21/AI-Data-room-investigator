import { Box, Checkbox, Collapse, IconButton, Typography } from "@mui/material";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import FolderOutlinedIcon from "@mui/icons-material/FolderOutlined";
import InsertDriveFileOutlinedIcon from "@mui/icons-material/InsertDriveFileOutlined";
import { filesUnder, type TreeNode } from "../../utils/fileTree";

interface Props {
  nodes: TreeNode[];
  selected: Set<string>;
  expanded: Set<string>;
  disabled?: boolean;
  onToggleFiles: (paths: string[], checked: boolean) => void;
  onToggleExpanded: (folder: string) => void;
}

function Row({ node, depth, ...rest }: Props & { node: TreeNode; depth: number }) {
  const { selected, expanded, disabled, onToggleFiles, onToggleExpanded } = rest;
  const files = filesUnder(node);
  const chosen = files.filter((f) => selected.has(f)).length;
  const checked = files.length > 0 && chosen === files.length;
  const open = expanded.has(node.path);

  return (
    <Box>
      <Box sx={{ display: "flex", alignItems: "center", pl: depth * 2.5 }}>
        {node.isFile ? (
          <Box sx={{ width: 34 }} />
        ) : (
          <IconButton size="small" onClick={() => onToggleExpanded(node.path)} aria-label={open ? "Collapse folder" : "Expand folder"}>
            {open ? <ExpandMoreIcon fontSize="small" /> : <ChevronRightIcon fontSize="small" />}
          </IconButton>
        )}
        <Checkbox
          size="small"
          checked={checked}
          indeterminate={chosen > 0 && !checked}
          disabled={disabled}
          onChange={(e) => onToggleFiles(files, e.target.checked)}
          slotProps={{ input: { "aria-label": node.isFile ? `Include ${node.path}` : `Include everything in ${node.path}` } }}
        />
        {node.isFile ? <InsertDriveFileOutlinedIcon fontSize="small" sx={{ mr: 1 }} /> : <FolderOutlinedIcon fontSize="small" sx={{ mr: 1 }} />}
        <Typography variant="body2" noWrap title={node.path}>{node.name}</Typography>
        {!node.isFile && (
          <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>{chosen}/{files.length}</Typography>
        )}
      </Box>
      {!node.isFile && (
        <Collapse in={open} unmountOnExit>
          {node.children.map((child) => (
            <Row key={child.path + String(child.isFile)} node={child} depth={depth + 1} {...rest} />
          ))}
        </Collapse>
      )}
    </Box>
  );
}

/** A folder hierarchy with a checkbox on every file and tri-state checkboxes on folders. */
export default function FileTree(props: Props) {
  return (
    <Box>
      {props.nodes.map((node) => (
        <Row key={node.path + String(node.isFile)} node={node} depth={0} {...props} />
      ))}
    </Box>
  );
}
