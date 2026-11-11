export interface TreeNode {
  name: string;
  path: string; // folder path or file path, relative to the data room
  isFile: boolean;
  children: TreeNode[];
}

/** Builds a folder hierarchy from relative file paths ("a/b/c.pdf"). Folders first, then files, each sorted by name. */
export function buildTree(paths: string[]): TreeNode[] {
  const root: TreeNode = { name: "", path: "", isFile: false, children: [] };
  for (const path of paths) {
    const parts = path.split("/");
    let node = root;
    parts.forEach((part, i) => {
      const isFile = i === parts.length - 1;
      const nodePath = parts.slice(0, i + 1).join("/");
      let child = node.children.find((c) => c.name === part && c.isFile === isFile);
      if (!child) {
        child = { name: part, path: nodePath, isFile, children: [] };
        node.children.push(child);
      }
      node = child;
    });
  }
  const sort = (nodes: TreeNode[]): TreeNode[] =>
    nodes
      .sort((a, b) => Number(a.isFile) - Number(b.isFile) || a.name.localeCompare(b.name))
      .map((n) => ({ ...n, children: sort(n.children) }));
  return sort(root.children);
}

/** Every file path at or below a node. */
export function filesUnder(node: TreeNode): string[] {
  return node.isFile ? [node.path] : node.children.flatMap(filesUnder);
}

export function allFolders(nodes: TreeNode[]): string[] {
  return nodes.flatMap((n) => (n.isFile ? [] : [n.path, ...allFolders(n.children)]));
}
