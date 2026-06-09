/* ═══════════════════════════════════════════════════════════
   PERA AI — Safe Markdown Renderer
   Converts a minimal markdown subset to React elements.
   NO dangerouslySetInnerHTML — fully structured rendering.
   ═══════════════════════════════════════════════════════════ */

import { createElement, type ReactNode } from "react";

type InlineSegment =
  | { type: "text"; value: string }
  | { type: "bold"; value: string }
  | { type: "italic"; value: string }
  | { type: "cite"; value: string };

/** Parse inline formatting: **bold**, *italic*, [N] citations */
function parseInline(text: string): InlineSegment[] {
  const segments: InlineSegment[] = [];
  // Regex alternation: bold first, then italic, then citations
  const re = /\*\*(.+?)\*\*|\*(.+?)\*|\[(\d+)\]/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = re.exec(text)) !== null) {
    if (match.index > lastIndex) {
      segments.push({ type: "text", value: text.slice(lastIndex, match.index) });
    }
    if (match[1] !== undefined) {
      segments.push({ type: "bold", value: match[1] });
    } else if (match[2] !== undefined) {
      segments.push({ type: "italic", value: match[2] });
    } else if (match[3] !== undefined) {
      segments.push({ type: "cite", value: match[3] });
    }
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    segments.push({ type: "text", value: text.slice(lastIndex) });
  }

  return segments;
}

export interface MarkdownOptions {
  /** Called when a clickable citation badge is activated. */
  onCite?: (n: number) => void;
  /** Citation numbers that have a matching source chip — only these become clickable. */
  citeSet?: Set<number>;
}

/** Render inline segments to React nodes */
function renderInline(
  text: string,
  keyPrefix: string,
  opts?: MarkdownOptions,
): ReactNode[] {
  return parseInline(text).map((seg, i) => {
    const key = `${keyPrefix}-${i}`;
    switch (seg.type) {
      case "bold":
        return createElement("strong", { key, className: "msg-inline-bold" }, seg.value);
      case "italic":
        return createElement("em", { key, className: "msg-inline-italic" }, seg.value);
      case "cite": {
        const n = parseInt(seg.value, 10);
        if (opts?.onCite && opts.citeSet?.has(n)) {
          return createElement(
            "button",
            {
              key,
              type: "button",
              className: "cite-badge cite-badge-link",
              onClick: () => opts.onCite!(n),
              title: `Jump to source [${n}]`,
              "aria-label": `Jump to source ${n}`,
            },
            `[${seg.value}]`,
          );
        }
        return createElement("span", { key, className: "cite-badge" }, `[${seg.value}]`);
      }
      default:
        return createElement("span", { key }, seg.value);
    }
  });
}

interface BlockElement {
  type: "heading" | "paragraph" | "ul" | "ol" | "spacer";
  level?: number;        // heading level (2-4)
  content?: string;      // for heading/paragraph
  items?: string[];      // for lists
}

/** Parse markdown text into block-level elements */
function parseBlocks(text: string): BlockElement[] {
  const rawLines = text.split("\n");
  const blocks: BlockElement[] = [];
  let listBuffer: { type: "ul" | "ol"; items: string[] } | null = null;

  const flushList = () => {
    if (!listBuffer) return;
    blocks.push({ type: listBuffer.type, items: listBuffer.items });
    listBuffer = null;
  };

  for (let i = 0; i < rawLines.length; i++) {
    const trimmed = rawLines[i].trim();

    if (!trimmed) {
      flushList();
      if (i > 0 && rawLines[i - 1]?.trim()) {
        blocks.push({ type: "spacer" });
      }
      continue;
    }

    // Headings: ####, ###, ##
    const headingMatch = trimmed.match(/^(#{2,4})\s+(.+)$/);
    if (headingMatch) {
      flushList();
      blocks.push({
        type: "heading",
        level: headingMatch[1].length,
        content: headingMatch[2],
      });
      continue;
    }

    // Bullet list
    if (/^[-•]\s/.test(trimmed)) {
      const content = trimmed.replace(/^[-•]\s*/, "");
      if (listBuffer?.type === "ul") {
        listBuffer.items.push(content);
      } else {
        flushList();
        listBuffer = { type: "ul", items: [content] };
      }
      continue;
    }

    // Numbered list
    if (/^\d+[.)]\s/.test(trimmed)) {
      const content = trimmed.replace(/^\d+[.)]\s*/, "");
      if (listBuffer?.type === "ol") {
        listBuffer.items.push(content);
      } else {
        flushList();
        listBuffer = { type: "ol", items: [content] };
      }
      continue;
    }

    // Normal paragraph
    flushList();
    blocks.push({ type: "paragraph", content: trimmed });
  }

  flushList();
  return blocks;
}

/**
 * Sanitize PARTIAL markdown emitted mid-stream so the user never sees raw,
 * incomplete tokens (dangling `**`, bare/partial `#` heading markers).
 * Completed pairs, citations `[n]`, numbered/bullet list markers, and line
 * breaks are all preserved untouched. Safe to call every streaming frame.
 */
export function sanitizeStreamingMarkdown(text: string): string {
  if (!text) return text;

  const lines = text.split("\n").map((line) => {
    // Heading WITH content streamed: normalize the hash count to the range the
    // block renderer supports (2–4) so a single `#` or `#####` still renders as
    // a heading instead of leaking raw `#`.
    const withContent = line.match(/^(\s*)(#{1,6})\s+(.*\S.*)$/);
    if (withContent) {
      const indent = withContent[1];
      const level = Math.min(Math.max(withContent[2].length, 2), 4);
      return `${indent}${"#".repeat(level)} ${withContent[3]}`;
    }
    // Bare/partial heading markers (hashes typed, heading text not streamed
    // yet) → keep only the indentation so raw `#`/`##`/`###` never flashes.
    if (/^\s*#{1,6}\s*$/.test(line)) {
      return (line.match(/^(\s*)/) as RegExpMatchArray)[1];
    }
    return line;
  });

  let result = lines.join("\n");

  // Dangling bold marker: an odd count of `**` means the closing pair hasn't
  // streamed yet. Drop the LAST unmatched `**` only — every completed
  // `**bold**` pair stays intact, and the closer renders bold once it arrives.
  if ((result.match(/\*\*/g) || []).length % 2 === 1) {
    const idx = result.lastIndexOf("**");
    result = result.slice(0, idx) + result.slice(idx + 2);
  }

  return result;
}

/** Render a full markdown string to an array of React elements — fully safe, no raw HTML */
export function renderMarkdown(text: string, opts?: MarkdownOptions): ReactNode[] {
  const blocks = parseBlocks(text);

  return blocks.map((block, i) => {
    const key = `b-${i}`;

    switch (block.type) {
      case "heading": {
        const tag = block.level === 2 ? "h3" : block.level === 3 ? "h4" : "h5";
        return createElement(
          tag,
          { key, className: "msg-heading" },
          ...renderInline(block.content!, key, opts),
        );
      }

      case "paragraph":
        return createElement(
          "p",
          { key, className: "msg-para" },
          ...renderInline(block.content!, key, opts),
        );

      case "ul":
        return createElement(
          "ul",
          { key, className: "msg-list msg-list-ul" },
          block.items!.map((item, j) =>
            createElement(
              "li",
              { key: `${key}-li-${j}` },
              ...renderInline(item, `${key}-li-${j}`, opts),
            ),
          ),
        );

      case "ol":
        return createElement(
          "ol",
          { key, className: "msg-list msg-list-ol" },
          block.items!.map((item, j) =>
            createElement(
              "li",
              { key: `${key}-li-${j}` },
              ...renderInline(item, `${key}-li-${j}`, opts),
            ),
          ),
        );

      case "spacer":
        return createElement("div", { key, className: "h-2" });

      default:
        return null;
    }
  });
}
