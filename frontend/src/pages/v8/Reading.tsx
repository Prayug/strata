/* How a question was read, as one sentence whose important parts can be changed in place (V10 brief, section 5):
     On [SPY], when [the previous session fell 1% or more]: how often it closes above its price at 10:00 New York time,
     [over the whole stored history].
   A click on a part lists its changes under the sentence. Choosing one sends it as a follow-up through the same reader as anything typed,
   so the change is said back in words and can be undone. Nothing here computes or guesses: the parts come from the server. */
import { useEffect, useState } from "react";

export type Part = { t: "text" | "ticker" | "period" | "condition"; s: string; choices?: { label: string; say: string }[]; default?: boolean };

const NAME: Record<string, string> = { ticker: "the ticker", period: "the period", condition: "this condition" };

/* The changes one part offers, as a row under the sentence (not a floating menu: under a part that wraps over two lines on
   a phone, a floating menu ran off the screen). */
function Choices({ part, onPick, onClose }: { part: Part; onPick: (say: string) => void; onClose: () => void }) {
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
  return (
    <span role="group" aria-label={`Change ${NAME[part.t] || "this"}`} className="mt-2 flex flex-wrap items-center gap-2" data-testid="reading-menu"
      style={{ fontFamily: "var(--font-sans)", fontSize: 13.5, lineHeight: 1.4 }}>
      <span className="mute">Change {NAME[part.t] || "this"}:</span>
      {(part.choices || []).map((c, i) => (
        <button key={i} className="act" style={{ height: 28, padding: "0 10px", fontSize: 13 }} onClick={() => onPick(c.say)} data-testid="reading-choice">{c.label}</button>
      ))}
      <button className="link" style={{ fontSize: 12.5 }} onClick={onClose}>cancel</button>
    </span>
  );
}

/** `live`: the parts can be clicked (the newest reading of a notebook). An older reading is the same sentence as text. */
export function ReadingLine({ parts, text, live, onSay }: { parts?: Part[]; text?: string; live: boolean; onSay: (t: string) => void }) {
  const [open, setOpen] = useState<number | null>(null);
  if (!parts || !parts.length) return <span data-testid="reading-line">{text}</span>;
  return (
    <span data-testid="reading-line">
      {parts.map((p, i) => {
        const can = live && p.t !== "text" && (p.choices || []).length > 0;
        if (!can) return <span key={i}>{p.s}</span>;
        return (
          /* a span, not a button: a long condition has to wrap with the sentence, and a button cannot break across lines */
          <span key={i} role="button" tabIndex={0} className="xpart" aria-expanded={open === i} data-testid={`reading-${p.t}`} title={`Change ${NAME[p.t] || "this"}`}
            onClick={() => setOpen(open === i ? null : i)} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setOpen(open === i ? null : i); } }}>{p.s}</span>
        );
      })}
      {open !== null && live && parts[open] && <Choices part={parts[open]} onClose={() => setOpen(null)} onPick={(say) => { setOpen(null); onSay(say); }} />}
    </span>
  );
}
