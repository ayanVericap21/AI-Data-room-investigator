import { useLayoutEffect, useRef, useState } from "react";
import { useWindowVirtualizer } from "@tanstack/react-virtual";

/**
 * Renders only the rows of a long list that are on screen (plus a few either side). The page itself scrolls, so the window is the scroll
 * container. Attach `ref` to the element that holds the rows, give each rendered row `ref={measure}` and `data-index={item.index}`, and leave
 * `before` / `after` pixels of space above and below them. Rows may differ in height: each one is measured once it is shown.
 * `count` 0 turns it off (a list on a tab that is not open).
 */
export function useVirtualRows(count: number, estimatedRowHeight: number, getKey?: (index: number) => string | number, overscan = 6) {
  const ref = useRef<HTMLDivElement>(null);
  const [margin, setMargin] = useState(0); // distance from the top of the page to the first row; changes when things above it open or close
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const top = Math.round(el.getBoundingClientRect().top + window.scrollY);
    setMargin((m) => (Math.abs(m - top) < 1 ? m : top));
  });
  const virtualizer = useWindowVirtualizer({
    count,
    estimateSize: () => estimatedRowHeight,
    overscan,
    scrollMargin: margin,
    getItemKey: getKey,
  });
  const items = virtualizer.getVirtualItems();
  const before = items.length ? items[0].start - margin : 0;
  const after = items.length ? virtualizer.getTotalSize() - (items[items.length - 1].end - margin) : 0;
  return { ref, items, before: Math.max(0, before), after: Math.max(0, after), measure: virtualizer.measureElement };
}
