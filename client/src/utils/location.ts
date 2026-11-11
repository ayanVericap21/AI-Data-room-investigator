import type { SourceRef } from "../types/claim";

/** Where a passage comes from: "p. 11", "Sheet!B3", or a heading. */
export function locationLabel(s: SourceRef): string {
  if (s.sheet_name) return `${s.sheet_name}!${s.cell_reference ?? ""}`;
  const parts: string[] = [];
  if (s.page_number) parts.push(`p. ${s.page_number}`);
  if (s.heading) parts.push(s.heading);
  return parts.join(" · ") || (s.content_type ?? "");
}
