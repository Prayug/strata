import { useMutation, useQuery } from "@tanstack/react-query";
import { Check, KeyRound, Loader2, ShieldCheck, X } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { ErrorBox, Loading, PageHeader, Panel } from "@/components/common";
import { Button, Field, Input, Select } from "@/components/ui";
import { api, queryClient } from "@/lib/api";
import { cn } from "@/lib/format";

const LLM_KEYS: Record<string, string> = { anthropic: "ANTHROPIC_API_KEY", openai: "OPENAI_API_KEY", gemini: "GEMINI_API_KEY", openrouter: "OPENROUTER_API_KEY" };
const DATA_KEYS = ["ALPACA_API_KEY", "ALPACA_API_SECRET", "POLYGON_API_KEY", "DATABENTO_API_KEY", "ORATS_API_KEY"];
const ZONES = [{ value: "America/Los_Angeles", label: "Pacific" }, { value: "America/Denver", label: "Mountain" }, { value: "America/Chicago", label: "Central" }, { value: "America/New_York", label: "Eastern" }];

export default function Settings() {
  const q = useQuery({ queryKey: ["settings"], queryFn: () => api.get<any>("/api/settings") });
  const ollama = useQuery({ queryKey: ["ollama"], queryFn: () => api.get<{ models: string[] }>("/api/settings/ollama"), staleTime: 60_000 });
  const [draft, setDraft] = useState<Record<string, any>>({});
  const [keys, setKeys] = useState<Record<string, string>>({});
  const [tests, setTests] = useState<Record<string, any>>({});
  useEffect(() => { if (q.data) setDraft(q.data.settings); }, [q.data]);
  const save = useMutation({
    mutationFn: (body: any) => api.put<any>("/api/settings", body),
    onSuccess: (d) => { queryClient.setQueryData(["settings"], d); queryClient.invalidateQueries({ queryKey: ["ai-meta"] }); queryClient.invalidateQueries({ queryKey: ["sources"] }); setKeys({}); toast.success("Settings saved"); },
    onError: (e: any) => toast.error(e.message),
  });
  const test = useMutation({
    mutationFn: async (provider: string) => ({ provider, r: await api.post<any>("/api/settings/llm/test", { provider }) }),
    onSuccess: ({ provider, r }) => setTests((t) => ({ ...t, [provider]: r })),
  });
  if (q.isLoading) return <Loading />;
  if (q.error) return <ErrorBox className="m-3" error={q.error} />;
  const d = q.data;
  const set = (k: string, v: any) => setDraft((x) => ({ ...x, [k]: v }));
  const dirty = JSON.stringify(draft) !== JSON.stringify(d.settings) || Object.values(keys).some((v) => v !== undefined && v !== "");
  const commit = () => save.mutate({ settings: draft, secrets: Object.fromEntries(Object.entries(keys).filter(([, v]) => v !== "")) });
  const clearKey = (name: string) => confirm(`Remove the stored ${d.secrets[name].label} key?`) && save.mutate({ settings: {}, secrets: { [name]: "" } });
  const keyRow = (name: string) => {
    const s = d.secrets[name];
    return (
      <div key={name} className="flex items-center gap-2">
        <span className="w-36 shrink-0 text-xs text-fg-2">{s.label}</span>
        <Input sz="sm" type="password" autoComplete="off" className="num flex-1" placeholder={s.set ? `${s.hint || "set"}  (${s.source})` : "not set"} value={keys[name] || ""} onChange={(e) => setKeys((k) => ({ ...k, [name]: e.target.value }))} />
        {s.source === "settings" && <Button size="iconSm" variant="ghost" title="Remove stored key" onClick={() => clearKey(name)}><X /></Button>}
        <span className={cn("w-4", s.set ? "text-up" : "text-faint")}>{s.set ? <Check className="size-3.5" /> : null}</span>
      </div>
    );
  };
  return (
    <div className="flex h-full flex-col">
      <PageHeader title="Settings"><Button variant="primary" disabled={!dirty || save.isPending} onClick={commit}>{save.isPending ? <Loader2 className="animate-spin" /> : <Check />} Save changes</Button></PageHeader>
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto grid max-w-[1100px] grid-cols-2 gap-3 p-4">
          <Panel title="AI interpreter" className="col-span-2">
            <div className="grid grid-cols-[1fr_1.2fr] gap-4 p-3">
              <div className="flex flex-col gap-2.5">
                <Field label="Who reads your requests" hint="The interpreter only turns your words into a structured draft. The engine computes every result.">
                  <Select value={draft["llm.provider"] || "auto"} onChange={(v) => set("llm.provider", v)} options={[
                    { value: "auto", label: "Automatic - first configured cloud model, else built-in" }, { value: "offline", label: "Built-in interpreter only (no model, no network)" },
                    ...d.llm.providers.map((p: any) => ({ value: p.id, label: p.label })),
                  ]} />
                </Field>
                <div className="rounded-sm border border-line bg-bg-0 px-2.5 py-2 text-xs">
                  <div className="label mb-0.5">Active now</div>
                  {d.llm.error ? <span className="text-warn">{d.llm.error} - falling back to the built-in interpreter</span> : <span className="text-fg">{d.llm.active || "Built-in rule-based interpreter"}</span>}
                </div>
                <p className="text-2xs leading-relaxed text-muted">
                  With no key, a built-in interpreter handles the common phrasings (entries, exits, previous-day moves, gaps, VWAP, streaks, sweeps, comparisons).
                  A language model accepts freer wording. Either way the model can only choose from a fixed list of tools; it cannot run code, write SQL or state a result.
                  Requests to Claude opt in to server-side refusal fallbacks, so a declined request is retried on another Claude model in the same call.
                </p>
              </div>
              <div className="flex flex-col gap-1.5">
                <div className="label">Providers</div>
                {d.llm.providers.map((p: any) => (
                  <div key={p.id} className="grid grid-cols-[110px_1fr_auto] items-center gap-2">
                    <span className="text-xs text-fg-2">{p.label}</span>
                    {p.id === "ollama"
                      ? <Select sz="sm" value={draft["llm.model.ollama"] || ""} onChange={(v) => set("llm.model.ollama", v)} options={[{ value: "", label: ollama.data?.models.length ? "first installed model" : "no local server found" }, ...(ollama.data?.models || []).map((m) => ({ value: m, label: m }))]} />
                      : <Input sz="sm" className="font-mono text-2xs" value={draft[`llm.model.${p.id}`] ?? ""} placeholder={p.default_model} onChange={(e) => set(`llm.model.${p.id}`, e.target.value)} />}
                    <div className="flex items-center gap-1.5">
                      <Button size="sm" variant="outline" disabled={test.isPending || (p.needs_key && !p.configured)} onClick={() => test.mutate(p.id)}>{test.isPending && test.variables === p.id ? <Loader2 className="animate-spin" /> : null} Test</Button>
                      <span className="w-40 truncate text-2xs" title={tests[p.id]?.error}>{tests[p.id] ? (tests[p.id].ok ? <span className="text-up">answered ({tests[p.id].model?.split(":").slice(1).join(":")})</span> : <span className="text-down">{tests[p.id].error}</span>) : p.needs_key ? (p.configured ? <span className="text-muted">key set</span> : <span className="text-faint">no key</span>) : <span className="text-muted">local</span>}</span>
                    </div>
                  </div>
                ))}
                <Field label="Ollama URL" className="mt-1"><Input sz="sm" className="font-mono text-2xs" value={draft["llm.ollama_url"] || ""} onChange={(e) => set("llm.ollama_url", e.target.value)} /></Field>
              </div>
            </div>
          </Panel>
          <Panel title={<span className="flex items-center gap-1.5"><KeyRound className="size-3" /> Model API keys</span>}>
            <div className="flex flex-col gap-1.5 p-3">{Object.values(LLM_KEYS).map(keyRow)}</div>
          </Panel>
          <Panel title={<span className="flex items-center gap-1.5"><KeyRound className="size-3" /> Data provider keys</span>}>
            <div className="flex flex-col gap-1.5 p-3">
              {DATA_KEYS.map(keyRow)}
              <p className="mt-1 text-2xs leading-relaxed text-muted">Yahoo, Cboe, DoltHub and the open option archive need no key. Alpaca's free account adds 1-minute bars since 2016. Paid providers are only ever called when you start a download yourself.</p>
            </div>
          </Panel>
          <Panel title="Preferences">
            <div className="grid grid-cols-2 gap-2.5 p-3">
              <Field label="Clock times without a timezone are" hint="“Buy at 10:15” is read in this timezone and converted to New York time. The interpretation panel always shows the converted time."><Select value={draft["ui.timezone"] || "America/Los_Angeles"} onChange={(v) => set("ui.timezone", v)} options={ZONES} /></Field>
              <Field label="Default symbol"><Input className="num uppercase" value={draft["data.default_symbol"] || ""} onChange={(e) => set("data.default_symbol", e.target.value.toUpperCase())} /></Field>
            </div>
          </Panel>
          <Panel title={<span className="flex items-center gap-1.5"><ShieldCheck className="size-3" /> How keys are handled</span>}>
            <ul className="list-inside list-disc space-y-1 p-3 text-2xs leading-relaxed text-muted">
              <li>Keys are stored in the local database on this machine (or read from environment variables) and used only by the backend.</li>
              <li>They are write-only here: the browser only ever receives the last four characters as a hint.</li>
              <li>Model requests contain your typed request, the current draft and the names and date ranges of stored datasets - never prices, trades or account details.</li>
              <li>Nothing that costs money is triggered automatically: the “Test” button sends one tiny request, and paid data sources never start on their own.</li>
            </ul>
          </Panel>
        </div>
      </div>
    </div>
  );
}
