/* The structured view of a strategy: a small code editor (a transparent textarea over a highlighted layer) with
   what an IDE needs here and nothing more - syntax colours for the document and for the rule language inside it,
   error markers on the line they belong to, a tint for the selected section and for lines an edit just changed,
   completion inside rules, and a clickable gutter (used for the debugger's breakpoint). No dependency. */
import { forwardRef, useEffect, useImperativeHandle, useLayoutEffect, useMemo, useRef, useState } from "react";
import { cn } from "@/lib/format";
import { lineOf } from "./jsonPath";

const LH = 19;   // line height, px
const PAD = 10;  // left padding of the text, px

export interface Mark { from: number; to: number; kind: "select" | "changed" }
export interface Diag { offset: number; message: string }
export interface Completion { from: number; to: number; items: { label: string; detail?: string; insert?: string; kind?: string }[] }
export interface EditorHandle { reveal: (offset: number) => void; focus: () => void }

const RULE_KEYS = new Set(["expr", "when", "break_when"]);

function ruleTokens(src: string, fns: Set<string>, names: Set<string>, key: number) {
  const out: React.ReactNode[] = [];
  const re = /\\.|'[^']*'|\b\d+(?:\.\d+)?\b|\b(?:and|or|not|True|False|None)\b|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*|[<>=!]=?|[()+\-*/,%]|\s+|./g;
  let m: RegExpExecArray | null, k = 0;
  while ((m = re.exec(src))) {
    const t = m[0];
    let c = "text-[#c9d1dc]";
    if (/^\d/.test(t)) c = "text-[#d19a66]";
    else if (/^(and|or|not|True|False|None)$/.test(t)) c = "text-[#c678dd]";
    else if (/^[A-Za-z_]/.test(t)) c = fns.has(t) ? "text-[#61afef]" : names.has(t) || names.has(t.split(".")[0]) || /^[A-Z]{1,5}\./.test(t) ? "text-[#56b6c2]" : "text-[#e5c07b]";
    else if (/^[<>=!]/.test(t)) c = "text-[#e06c75]";
    else if (t.startsWith("\\") || t.startsWith("'")) c = "text-[#98c379]";
    out.push(<span key={`${key}-${k++}`} className={c}>{t}</span>);
  }
  return out;
}

function highlight(text: string, fns: Set<string>, names: Set<string>) {
  const out: React.ReactNode[] = [];
  const re = /"(?:[^"\\\n]|\\.)*"?(\s*:)?|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\b(?:true|false|null)\b|[{}[\],:]|\s+|./g;
  let m: RegExpExecArray | null, k = 0, lastKey = "";
  while ((m = re.exec(text))) {
    const t = m[0];
    if (t[0] === '"') {
      if (m[1]) {
        const name = t.slice(0, t.length - m[1].length);
        lastKey = name.slice(1, -1);
        out.push(<span key={k++} className="text-[#8fa1b8]">{name}</span>, <span key={k++} className="text-faint">{m[1]}</span>);
      } else if (RULE_KEYS.has(lastKey) && t.length > 1 && t.endsWith('"')) {
        out.push(<span key={k++} className="text-faint">"</span>, <span key={k++} className="rounded-[2px] bg-accent/[0.06]">{ruleTokens(t.slice(1, -1), fns, names, k)}</span>, <span key={k++} className="text-faint">"</span>);
        k += 2;
      } else out.push(<span key={k++} className="text-[#98c379]">{t}</span>);
    } else if (/^-?\d/.test(t)) out.push(<span key={k++} className="text-[#d19a66]">{t}</span>);
    else if (/^(true|false|null)$/.test(t)) out.push(<span key={k++} className="text-[#c678dd]">{t}</span>);
    else if (/^[{}[\],:]$/.test(t)) out.push(<span key={k++} className="text-faint">{t}</span>);
    else out.push(t);
  }
  return out;
}

export const CodeEditor = forwardRef<EditorHandle, {
  value: string; onChange: (v: string) => void; onCaret?: (start: number, end: number) => void; marks?: Mark[]; diags?: Diag[]; breakLine?: number | null; onGutter?: (line: number) => void;
  complete?: (text: string, caret: number) => Completion | null; functions?: Set<string>; names?: Set<string>; readOnly?: boolean; ariaLabel?: string;
}>(({ value, onChange, onCaret, marks = [], diags = [], breakLine, onGutter, complete, functions, names, readOnly, ariaLabel }, ref) => {
  const ta = useRef<HTMLTextAreaElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const probe = useRef<HTMLSpanElement>(null);
  const [cw, setCw] = useState(7.2);
  const [pop, setPop] = useState<(Completion & { sel: number; top: number; left: number }) | null>(null);
  const fns = functions || EMPTY, nms = names || EMPTY;
  const lines = useMemo(() => value.split("\n").length, [value]);
  const html = useMemo(() => highlight(value, fns, nms), [value, fns, nms]);
  useLayoutEffect(() => { if (probe.current) setCw(probe.current.getBoundingClientRect().width / 40 || 7.2); }, []);
  useImperativeHandle(ref, () => ({
    reveal: (offset) => { const s = scroller.current; if (!s) return; const y = lineOf(value, offset) * LH; if (y < s.scrollTop + LH || y > s.scrollTop + s.clientHeight - 3 * LH) s.scrollTo({ top: Math.max(0, y - s.clientHeight / 3), behavior: "smooth" }); },
    focus: () => ta.current?.focus(),
  }), [value]);

  const lineMarks = useMemo(() => {
    const m = new Map<number, string>();
    for (const k of marks) {
      const a = lineOf(value, k.from), b = lineOf(value, Math.max(k.from, k.to - 1));
      for (let l = a; l <= b; l++) if (k.kind === "changed" || !m.has(l)) m.set(l, k.kind);
    }
    return m;
  }, [marks, value]);
  const lineDiags = useMemo(() => {
    const m = new Map<number, string[]>();
    for (const d of diags) { const l = lineOf(value, d.offset); m.set(l, [...(m.get(l) || []), d.message]); }
    return m;
  }, [diags, value]);

  const insert = (text: string, from: number, to: number) => {
    const el = ta.current!;
    el.focus();
    el.setSelectionRange(from, to);
    if (!document.execCommand?.("insertText", false, text)) { el.setRangeText(text, from, to, "end"); onChange(el.value); }
  };
  const refreshPop = () => {
    const el = ta.current;
    if (!el || !complete || el.selectionStart !== el.selectionEnd) { setPop(null); return; }
    const c = complete(el.value, el.selectionStart);
    if (!c || !c.items.length) { setPop(null); return; }
    const before = el.value.slice(0, c.from);
    const line = lineOf(el.value, c.from), col = c.from - (before.lastIndexOf("\n") + 1);
    setPop((p) => ({ ...c, sel: p && p.from === c.from ? Math.min(p.sel, c.items.length - 1) : 0, top: (line + 1) * LH + 6, left: PAD + 44 + col * cw }));
  };
  const accept = (i: number) => { if (!pop) return; const it = pop.items[i]; insert(it.insert ?? it.label, pop.from, pop.to); setPop(null); };
  const caret = () => { const el = ta.current; if (el) onCaret?.(el.selectionStart, el.selectionEnd); };
  useEffect(() => { if (pop) scroller.current?.querySelector<HTMLElement>("[data-active=true]")?.scrollIntoView({ block: "nearest" }); }, [pop?.sel]); // eslint-disable-line react-hooks/exhaustive-deps

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    const el = e.currentTarget;
    if (pop) {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); setPop({ ...pop, sel: (pop.sel + (e.key === "ArrowDown" ? 1 : pop.items.length - 1)) % pop.items.length }); return; }
      if (e.key === "Enter" || e.key === "Tab") { e.preventDefault(); accept(pop.sel); return; }
      if (e.key === "Escape") { e.preventDefault(); setPop(null); return; }
    }
    if (e.key === "Tab" && !e.metaKey && !e.ctrlKey && !e.altKey) { e.preventDefault(); insert("  ", el.selectionStart, el.selectionEnd); return; }
    if (e.key === "Enter" && !e.metaKey && !e.ctrlKey) {
      e.preventDefault();
      const before = el.value.slice(0, el.selectionStart);
      const ind = /^[ \t]*/.exec(before.slice(before.lastIndexOf("\n") + 1))![0];
      const opens = /[{[]\s*$/.test(before), closes = /^\s*[}\]]/.test(el.value.slice(el.selectionEnd));
      if (opens && closes) { const at = el.selectionStart; insert(`\n${ind}  \n${ind}`, at, el.selectionEnd); el.setSelectionRange(at + ind.length + 3, at + ind.length + 3); }
      else insert(`\n${ind}${opens ? "  " : ""}`, el.selectionStart, el.selectionEnd);
    }
  };

  return (
    <div ref={scroller} className="relative h-full min-h-0 overflow-auto bg-bg-0 font-mono text-[12px]" style={{ lineHeight: `${LH}px` }} data-testid="code-editor">
      <span ref={probe} className="pointer-events-none invisible absolute whitespace-pre">{"0".repeat(40)}</span>
      <div className="relative flex min-h-full min-w-max">
        <div className="sticky left-0 z-[2] w-[44px] shrink-0 select-none border-r border-line bg-bg-1 text-right text-[10.5px] text-faint" aria-hidden>
          {Array.from({ length: lines }, (_, l) => {
            const d = lineDiags.get(l);
            return (
              <div key={l} onClick={() => onGutter?.(l)} title={d ? d.join("\n") : onGutter ? "Click to break in the debugger when this line's rule first holds" : undefined}
                className={cn("relative h-[19px] cursor-pointer pr-2 hover:text-fg", d && "bg-down/20 font-semibold text-down", breakLine === l && "text-accent")}>
                {breakLine === l && <span className="absolute left-1 top-[5px] size-[9px] rounded-full bg-accent" />}{d ? "!" : l + 1}
              </div>
            );
          })}
        </div>
        <div className="relative min-w-0 flex-1">
          {[...lineMarks].map(([l, kind]) => <div key={`m${l}`} className={cn("pointer-events-none absolute inset-x-0", kind === "changed" ? "bg-up/[0.13] shadow-[inset_2px_0_0_var(--color-up)]" : "bg-accent/[0.07] shadow-[inset_2px_0_0_var(--color-accent)]")} style={{ top: l * LH, height: LH }} />)}
          {[...lineDiags.keys()].map((l) => <div key={`d${l}`} className="pointer-events-none absolute inset-x-0 border-b border-dashed border-down/70 bg-down/[0.08]" style={{ top: l * LH, height: LH }} />)}
          <pre className="pointer-events-none relative m-0 whitespace-pre pb-24 pr-10 text-fg" style={{ paddingLeft: PAD, tabSize: 2 }} aria-hidden>{html}{"\n"}</pre>
          <textarea ref={ta} value={value} readOnly={readOnly} spellCheck={false} autoCapitalize="off" autoCorrect="off" wrap="off" aria-label={ariaLabel || "Strategy source"}
            className="absolute inset-0 m-0 resize-none overflow-hidden whitespace-pre border-0 bg-transparent pb-24 pr-10 text-transparent caret-[var(--color-accent-2)] outline-none selection:bg-accent/30"
            style={{ paddingLeft: PAD, lineHeight: `${LH}px`, tabSize: 2, fontFamily: "inherit", fontSize: "inherit" }}
            onChange={(e) => { onChange(e.target.value); requestAnimationFrame(() => { refreshPop(); caret(); }); }}
            onKeyDown={onKeyDown} onKeyUp={(e) => { if (!["ArrowDown", "ArrowUp", "Enter", "Tab", "Escape"].includes(e.key)) refreshPop(); caret(); }} onClick={() => { setPop(null); caret(); }} onSelect={caret} onBlur={() => setTimeout(() => setPop(null), 120)} />
          {pop && (
            <div className="absolute z-20 max-h-[212px] w-[420px] overflow-auto rounded-[3px] border border-line-2 bg-bg-2 py-0.5 shadow-xl" style={{ top: pop.top, left: Math.max(0, pop.left - 44) }} role="listbox" data-testid="completions">
              {pop.items.map((it, i) => (
                <div key={it.label} role="option" aria-selected={i === pop.sel} data-active={i === pop.sel} onMouseDown={(e) => { e.preventDefault(); accept(i); }}
                  className={cn("flex cursor-pointer items-baseline gap-2 px-2 py-[2px]", i === pop.sel ? "bg-accent/20 text-fg" : "text-fg-2 hover:bg-bg-3")}>
                  <span className="w-3 shrink-0 text-[10px] text-faint">{it.kind === "function" ? "ƒ" : it.kind === "field" ? "·" : it.kind === "key" ? "k" : "x"}</span>
                  <span className="shrink-0">{it.label}</span><span className="min-w-0 truncate font-sans text-[11px] text-muted">{it.detail}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
});
CodeEditor.displayName = "CodeEditor";
const EMPTY = new Set<string>();
