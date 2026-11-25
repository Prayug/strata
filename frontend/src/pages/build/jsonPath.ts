/* JSON with positions: the structured view needs to know which piece of text is which part of the strategy, so a
   section can be highlighted, an error can be put on its line, and the caret can say what is selected. Paths use
   the same spelling as the backend: entry.conditions.children[0].params.expr */

export interface Range { start: number; end: number; valueStart: number }
export type Parsed = { ok: true; value: any; ranges: Map<string, Range> } | { ok: false; error: string; offset: number };

export function parseRanges(text: string): Parsed {
  let i = 0;
  const ranges = new Map<string, Range>();
  const fail = (msg: string): never => { throw Object.assign(new Error(msg), { offset: i }); };
  const ws = () => { while (i < text.length && /\s/.test(text[i])) i++; };
  const str = (): string => {
    const a = i;
    i++;
    while (i < text.length && text[i] !== '"') { if (text[i] === "\\") i++; if (text[i] === "\n") fail("a string is not closed before the end of the line"); i++; }
    if (i >= text.length) fail("a string is not closed");
    i++;
    try { return JSON.parse(text.slice(a, i)); } catch { i = a; return fail("this string has an invalid escape"); }
  };
  const value = (path: string): any => {
    ws();
    const c = text[i];
    if (c === "{") {
      i++;
      const out: any = {};
      ws();
      if (text[i] === "}") { i++; return out; }
      for (;;) {
        ws();
        if (text[i] !== '"') fail(text[i] === "}" ? "remove the comma before this closing brace" : "a field name in double quotes is expected here");
        const ks = i;
        const k = str();
        ws();
        if (text[i] !== ":") fail(`a colon is expected after "${k}"`);
        i++;
        ws();
        const p = path ? `${path}.${k}` : k;
        const vs = i;
        out[k] = value(p);
        ranges.set(p, { start: ks, end: i, valueStart: vs });
        ws();
        if (text[i] === ",") { i++; continue; }
        if (text[i] === "}") { i++; return out; }
        fail("a comma or a closing brace is expected here");
      }
    }
    if (c === "[") {
      i++;
      const out: any[] = [];
      ws();
      if (text[i] === "]") { i++; return out; }
      for (;;) {
        ws();
        const p = `${path}[${out.length}]`;
        const vs = i;
        out.push(value(p));
        ranges.set(p, { start: vs, end: i, valueStart: vs });
        ws();
        if (text[i] === ",") { i++; continue; }
        if (text[i] === "]") { i++; return out; }
        fail("a comma or a closing bracket is expected here");
      }
    }
    if (c === '"') return str();
    const m = /^(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)/.exec(text.slice(i, i + 40));
    if (!m) fail(i >= text.length ? "the document ends before it is complete" : `"${text.slice(i, i + 12).split(/\s/)[0]}" is not a value (text needs double quotes)`);
    i += m![0].length;
    return JSON.parse(m![0]);
  };
  try {
    const v = value("");
    ws();
    if (i < text.length) fail("nothing is expected after the closing brace");
    if (typeof v !== "object" || v === null || Array.isArray(v)) { i = 0; fail("a strategy is an object: start with {"); }
    return { ok: true, value: v, ranges };
  } catch (e: any) {
    return { ok: false, error: e.message, offset: Math.min(e.offset ?? i, text.length) };
  }
}

/** The innermost path whose text contains the offset ("" for the document itself). */
export function pathAt(ranges: Map<string, Range>, offset: number): string {
  let best = "", size = Infinity;
  for (const [p, r] of ranges) if (offset >= r.start && offset <= r.end && r.end - r.start < size) { best = p; size = r.end - r.start; }
  return best;
}

export const lineOf = (text: string, offset: number) => { let n = 0; for (let k = 0; k < offset && k < text.length; k++) if (text.charCodeAt(k) === 10) n++; return n; };

/** Pretty JSON that keeps small leaves on one line, so a leg or an exit reads like a line of code. */
export function format(v: any, width = 100): string {
  const one = (x: any): string => (x === null || typeof x !== "object" ? JSON.stringify(x)
    : Array.isArray(x) ? `[${x.map(one).join(", ")}]` : `{ ${Object.entries(x).map(([k, y]) => `${JSON.stringify(k)}: ${one(y)}`).join(", ")} }`);
  const depth = (x: any): number => (x === null || typeof x !== "object" ? 0 : 1 + Math.max(0, ...Object.values(x).map(depth)));
  // ind: indentation of this value's own lines; col: the column its first character sits at (for the width test)
  const go = (x: any, ind: number, col: number): string => {
    if (x === null || typeof x !== "object") return JSON.stringify(x);
    if (Array.isArray(x) ? !x.length : !Object.keys(x).length) return Array.isArray(x) ? "[]" : "{}";
    const flat = one(x);
    if (depth(x) <= 2 && flat.length + col <= width && !("children" in x) && !("legs" in x)) return flat;
    const pad = " ".repeat(ind + 2);
    if (Array.isArray(x)) return `[\n${x.map((y) => pad + go(y, ind + 2, ind + 2)).join(",\n")}\n${" ".repeat(ind)}]`;
    return `{\n${Object.entries(x).map(([k, y]) => `${pad}${JSON.stringify(k)}: ${go(y, ind + 2, ind + 2 + k.length + 4)}`).join(",\n")}\n${" ".repeat(ind)}}`;
  };
  return go(v, 0, 0);
}

export function getAt(doc: any, path: string): any {
  let cur = doc;
  for (const t of tokens(path)) { if (cur === null || cur === undefined) return undefined; cur = cur[t as any]; }
  return cur;
}
export function setAt(doc: any, path: string, value: any): any {
  const out = JSON.parse(JSON.stringify(doc));
  const ts = tokens(path);
  let cur = out;
  ts.slice(0, -1).forEach((t, k) => { if (cur[t as any] === undefined || cur[t as any] === null) cur[t as any] = typeof ts[k + 1] === "number" ? [] : {}; cur = cur[t as any]; });
  if (value === undefined) { if (Array.isArray(cur)) cur.splice(ts[ts.length - 1] as number, 1); else delete cur[ts[ts.length - 1] as any]; }
  else cur[ts[ts.length - 1] as any] = value;
  return out;
}
export function tokens(path: string): (string | number)[] {
  const out: (string | number)[] = [];
  for (const part of path.split(".").filter(Boolean)) {
    const m = /^([^[\]]+)((?:\[\d+\])*)$/.exec(part);
    if (!m) continue;
    out.push(m[1]);
    for (const x of m[2].matchAll(/\[(\d+)\]/g)) out.push(Number(x[1]));
  }
  return out;
}
export const parentPath = (path: string) => path.replace(/(\.[^.[\]]+|\[\d+\])$/, "");
