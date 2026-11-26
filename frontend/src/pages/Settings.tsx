import { useMutation, useQuery } from "@tanstack/react-query";
import { Check, ExternalLink, KeyRound, Loader2, PlugZap, ShieldCheck, X } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { ErrorBox, Loading, PageHeader, Panel } from "@/components/common";
import { Button, Field, Input, Select } from "@/components/ui";
import { api, queryClient, useJob } from "@/lib/api";
import { ago, cn } from "@/lib/format";

const LLM_KEYS: Record<string, string> = { anthropic: "ANTHROPIC_API_KEY", openai: "OPENAI_API_KEY", gemini: "GEMINI_API_KEY", openrouter: "OPENROUTER_API_KEY" };
const DATA_KEYS = ["POLYGON_API_KEY", "DATABENTO_API_KEY", "ORATS_API_KEY"];
const ALPACA_KEYS = ["ALPACA_API_KEY", "ALPACA_API_SECRET"];
const ZONES = [{ value: "America/Los_Angeles", label: "Pacific" }, { value: "America/Denver", label: "Mountain" }, { value: "America/Chicago", label: "Central" }, { value: "America/New_York", label: "Eastern" }];

/** How well requests are understood, measured on written sentences with known readings. The built-in reader can always be
    measured; the model-backed path only when a model is configured, and it is reported as not run until then. */
function EvalPanel() {
  const q = useQuery({ queryKey: ["nl-eval"], queryFn: () => api.get<any>("/api/ai/evaluation") });
  const meta = useQuery({ queryKey: ["ai-meta"], queryFn: () => api.get<any>("/api/ai/meta"), staleTime: 30_000 });
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId, (st) => { setJobId(null); queryClient.invalidateQueries({ queryKey: ["nl-eval"] }); if (st.status === "error") toast.error(st.error || "The evaluation did not finish"); });
  const run = useMutation({ mutationFn: (llm: boolean) => api.post<{ run_id: string }>(`/api/ai/evaluation?llm=${llm}`), onSuccess: (r) => setJobId(r.run_id), onError: (e: any) => toast.error(e.message) });
  const r = q.data || {};
  const model = meta.data?.llm?.active;
  const rows: [string, any, string][] = r.development ? [
    ["Development sentences", r.development, "The reader was corrected against these, so this says what was fixed."],
    ["Held-out sentences, first run", r.held_out_first_run, "Written afterwards and read once before anything was changed: the honest estimate for a sentence it has not seen."],
    ["Held-out sentences, now", r.held_out, "After fixing what the first run exposed. These sentences are no longer unseen."],
  ] : [];
  const busy = !!jobId || run.isPending;
  return (
    <Panel title="How well requests are understood" className="col-span-2">
      <div className="p-3 text-xs" data-testid="eval-panel">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-fg-2">150 written requests with known readings. A reading counts as wrong whether or not the reader said it was unsure; "silently wrong" is the kind that matters.</span>
          <span className="flex-1" />
          {r.run_at && <span className="text-muted">measured {ago(r.run_at)} · {r.interpreter}</span>}
          <Button size="sm" variant="outline" disabled={busy} onClick={() => run.mutate(false)} data-testid="eval-run">{busy ? <Loader2 className="animate-spin" /> : <Check />} Measure the built-in reader</Button>
          <Button size="sm" variant="outline" disabled={busy || !model} onClick={() => run.mutate(true)} title={model ? `Reads the same sentences with ${model}` : "No language model is configured"}>Measure with the configured model</Button>
        </div>
        {job && <div className="mt-2 text-muted">{job.message || "reading the sentences"}…</div>}
        {rows.length > 0 ? (
          <table className="tbl mt-2.5" data-testid="eval-table">
            <thead><tr><th>Set</th><th className="r">Sentences</th><th className="r">Read correctly</th><th className="r">Wrong, and said it was unsure</th><th className="r">Silently wrong</th><th>What it means</th></tr></thead>
            <tbody>{rows.map(([label, x, help]) => (
              <tr key={label}><td>{label}</td><td className="r num">{x.cases}</td><td className="r num">{x.correct} <span className="text-muted">({x.accuracy_pct}%)</span></td><td className="r num">{x.flagged}</td>
                <td className={cn("r num", x.silent > 0 ? "text-warn" : "text-up")}>{x.silent}{x.silent_pct !== undefined ? <span className="text-muted"> ({x.silent_pct}%)</span> : null}</td><td className="min-w-[280px] whitespace-normal text-muted">{help}</td></tr>
            ))}</tbody>
          </table>
        ) : <div className="mt-2 text-muted">{r.note || "Not measured on this installation yet."}</div>}
        <div className={cn("mt-2 rounded-sm border px-2.5 py-2", model ? "border-line" : "border-warn/40 bg-warn/5")} data-testid="eval-llm">
          {model ? <>A language model is configured ({model}). Its reading of the same sentences is measured separately and never mixed into the figures above.</>
            : <><span className="font-medium text-warn">Model-backed reading: NOT RUN.</span> No language model is configured, so nothing can be said about it. To measure it, add a key under <span className="text-fg">Model API keys</span> below (or start a local Ollama and choose it under AI interpreter), then press <span className="text-fg">Measure with the configured model</span>. It is never reported as passing without being run.</>}
        </div>
      </div>
    </Panel>
  );
}

export default function Settings() {
  const q = useQuery({ queryKey: ["settings"], queryFn: () => api.get<any>("/api/settings") });
  const acc = useQuery({ queryKey: ["alpaca-access"], queryFn: () => api.get<any>("/api/data/access/alpaca"), staleTime: 30_000 });
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
    onSuccess: ({ provider, r }) => { setTests((t) => ({ ...t, [provider]: r })); queryClient.invalidateQueries({ queryKey: ["llm-usage"] }); },
  });
  const usage = useQuery({ queryKey: ["llm-usage"], queryFn: () => api.get<any>("/api/settings/llm/usage"), staleTime: 15_000 });
  // model discovery: ask the provider which models this account can use
  const [models, setModels] = useState<Record<string, { ok: boolean; models: string[]; error?: string }>>({});
  const discover = useMutation({
    mutationFn: async (provider: string) => ({ provider, r: await api.get<any>(`/api/settings/llm/models?provider=${provider}`) }),
    onSuccess: ({ provider, r }) => setModels((m) => ({ ...m, [provider]: r })),
  });
  // guided Alpaca connection: save the two keys (write-only), then make one small request to each API
  const [alpaca, setAlpaca] = useState<any>(null);
  const connect = useMutation({
    mutationFn: async () => {
      const secrets = Object.fromEntries(ALPACA_KEYS.filter((k) => keys[k]).map((k) => [k, keys[k]]));
      if (Object.keys(secrets).length) queryClient.setQueryData(["settings"], await api.put<any>("/api/settings", { settings: {}, secrets }));
      return api.post<any>("/api/data/alpaca/test");
    },
    onSuccess: (r) => { setAlpaca(r); setKeys((k) => Object.fromEntries(Object.entries(k).filter(([n]) => !ALPACA_KEYS.includes(n)))); queryClient.invalidateQueries({ queryKey: ["sources"] }); queryClient.invalidateQueries({ queryKey: ["coverage"] }); },
    onError: (e: any) => setAlpaca({ ok: false, message: e.message }),
  });
  useEffect(() => { if (location.hash === "#alpaca") setTimeout(() => document.getElementById("alpaca")?.scrollIntoView({ block: "start" }), 150); }, [q.isSuccess]);
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
      <div key={name} className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
        <span className="w-36 shrink-0 text-xs text-fg-2">{s.label}</span>
        <Input sz="sm" type="password" autoComplete="off" className="num flex-1" placeholder={s.set ? `${s.hint || "set"}  (${s.source})` : "not set"} value={keys[name] || ""} onChange={(e) => setKeys((k) => ({ ...k, [name]: e.target.value }))} />
        {s.set && s.source !== "environment variable" && <Button size="iconSm" variant="ghost" title="Remove stored key" onClick={() => clearKey(name)}><X /></Button>}
        <span className={cn("w-4", s.set ? "text-up" : "text-faint")}>{s.set ? <Check className="size-3.5" /> : null}</span>
        {!s.set && s.elsewhere && <div className="basis-full pl-[152px] text-2xs leading-snug text-warn" data-testid={`elsewhere-${name}`}>{s.elsewhere.note}</div>}
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
                <div className="grid grid-cols-2 gap-2.5">
                  <Field label="If the model fails or is unavailable" hint="Either way you are told which interpreter read each request; a complex strategy is never silently handed to a weaker one.">
                    <Select value={draft["llm.fallback"] || "rules"} onChange={(v) => set("llm.fallback", v)} options={[{ value: "rules", label: "Use the built-in interpreter and say so" }, { value: "none", label: "Stop and report the failure" }]} />
                  </Field>
                  <Field label="Who reads a request" hint="Measured on sentences neither reader had seen: the built-in readers alone answered a different question without saying so 9 times in 34; with a local model reading as well and every difference asked, 2 to 3 times.">
                    <Select value={draft["llm.mode"] || "auto"} onChange={(v) => set("llm.mode", v)} options={[{ value: "auto", label: "Automatic: both must agree with a local model; a hosted model reads alone" },
                      { value: "check", label: "Both read; a difference is asked before anything runs" }, { value: "assist", label: "Built-in readers first, the model for what they flag" }, { value: "model", label: "The model reads everything" }]} />
                  </Field>
                  <Field label="How a model is asked" hint="The form is a short list of typed fields that a small local model can fill; every value is checked against your sentence and the expressions are written by Strata. The full planner gives a capable hosted model every tool.">
                    <Select value={draft["llm.planner"] || "auto"} onChange={(v) => set("llm.planner", v)} options={[{ value: "auto", label: "Automatic: the form for a local model, the full planner for a hosted one" }, { value: "form", label: "The form of typed fields" }, { value: "tools", label: "The full planner" }]} />
                  </Field>
                  <Field label="Context asked of a local model (tokens)" hint="Left to the server this is the model's maximum, which made a 6 GB model occupy 13 GB and answer in minutes on this machine."><Input sz="sm" className="num" inputMode="numeric" value={draft["llm.ollama.context_tokens"] ?? 12288} onChange={(e) => set("llm.ollama.context_tokens", Math.max(2048, Math.min(131072, Number(e.target.value) || 12288)))} /></Field>
                  <Field label="New research starts in" hint="Quick research runs straight away and lists every assumption. Review first waits for you to confirm the interpretation.">
                    <Select value={draft["ai.exec_mode"] || "quick"} onChange={(v) => set("ai.exec_mode", v)} options={[{ value: "quick", label: "Quick research" }, { value: "review", label: "Review before running" }]} />
                  </Field>
                  <Field label="Request timeout (seconds)"><Input sz="sm" className="num" inputMode="numeric" value={draft["llm.timeout_seconds"] ?? 90} onChange={(e) => set("llm.timeout_seconds", Math.max(5, Math.min(600, Number(e.target.value) || 90)))} /></Field>
                  <Field label="Retries on a transient error" hint="Rate limits and server errors are retried with a growing delay. Authentication and request errors are not."><Input sz="sm" className="num" inputMode="numeric" value={draft["llm.max_retries"] ?? 2} onChange={(e) => set("llm.max_retries", Math.max(0, Math.min(6, Number(e.target.value) || 0)))} /></Field>
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
                  <div key={p.id} className="grid grid-cols-[110px_1fr_auto] items-center gap-2" title={models[p.id]?.error}>
                    <span className="text-xs text-fg-2">{p.label}</span>
                    {p.id === "ollama"
                      ? <Select sz="sm" value={draft["llm.model.ollama"] || ""} onChange={(v) => set("llm.model.ollama", v)} options={[{ value: "", label: ollama.data?.models.length ? "first installed model" : "no local server found" }, ...(ollama.data?.models || []).map((m) => ({ value: m, label: m }))]} />
                      : <>
                          <Input sz="sm" className="font-mono text-2xs" list={`models-${p.id}`} value={draft[`llm.model.${p.id}`] ?? ""} placeholder={p.default_model} onChange={(e) => set(`llm.model.${p.id}`, e.target.value)} />
                          <datalist id={`models-${p.id}`}>{(models[p.id]?.models || []).map((m) => <option key={m} value={m} />)}</datalist>
                        </>}
                    <div className="flex items-center gap-1.5">
                      {p.id !== "ollama" && <Button size="sm" variant="ghost" title="Ask the provider which models this account can use" disabled={discover.isPending || (p.needs_key && !p.configured && p.id !== "openrouter")} onClick={() => discover.mutate(p.id)}>{discover.isPending && discover.variables === p.id ? <Loader2 className="animate-spin" /> : null}{models[p.id] ? (models[p.id].ok ? `${models[p.id].models.length} models` : "failed") : "List models"}</Button>}
                      <Button size="sm" variant="outline" disabled={test.isPending || (p.needs_key && !p.configured)} onClick={() => test.mutate(p.id)}>{test.isPending && test.variables === p.id ? <Loader2 className="animate-spin" /> : null} Test</Button>
                      <span className="w-40 truncate text-2xs" title={tests[p.id]?.error}>{tests[p.id] ? (tests[p.id].ok ? <span className="text-up">answered ({tests[p.id].model?.split(":").slice(1).join(":")})</span> : <span className="text-down">{tests[p.id].error}</span>) : p.needs_key ? (p.configured ? <span className="text-muted">key set</span> : <span className="text-faint">no key</span>) : <span className="text-muted">local</span>}</span>
                    </div>
                  </div>
                ))}
                <Field label="Ollama URL" className="mt-1"><Input sz="sm" className="font-mono text-2xs" value={draft["llm.ollama_url"] || ""} onChange={(e) => set("llm.ollama_url", e.target.value)} /></Field>
              </div>
            </div>
          </Panel>
          <EvalPanel />
          <Panel title={<span className="flex items-center gap-1.5"><PlugZap className="size-3" /> Connect a free Alpaca account</span>} className="col-span-2">
            <div id="alpaca" className="grid grid-cols-[1.1fr_1fr] gap-5 p-3">
              <div className="text-xs leading-relaxed text-fg-2">
                {acc.data?.verified ? (
                  <p className="text-muted" data-testid="alpaca-measured">Measured with the key stored here ({String(acc.data.run_at || "").slice(0, 10)}): {(acc.data.summary || []).filter((x: string) => /matrix|history starts|Stock minute/.test(x)).join(" ")} No bid/ask history exists on any Alpaca plan: its option data is trade prints. The full report is under <a className="text-info hover:underline" href="/data?tab=readiness">Data, Readiness</a>. Strata only reads market data with the key; it has no code that can place an order.</p>
                ) : (
                  <p className="text-muted">Alpaca documents, for its free account, 1-minute stock bars since 2016 and 1-minute option trade bars since early 2024 (no bid/ask history on any plan). That is the provider's description: what this account actually returns is established only by the access test under <a className="text-info hover:underline" href="/data?tab=readiness">Data, Readiness</a>, which stays BLOCKED until a key is stored here. Strata only reads market data with it; it has no code that can place an order.</p>
                )}
                <ol className="mt-2.5 flex list-inside list-decimal flex-col gap-1.5">
                  <li>Create a free account at <a className="text-info hover:underline" href="https://alpaca.markets" target="_blank" rel="noreferrer">alpaca.markets <ExternalLink className="inline size-3" /></a>. A paper-trading account is enough; no deposit is needed.</li>
                  <li>In the Alpaca dashboard, open <span className="text-fg">API Keys</span> and generate a key. You get a key ID and a secret.</li>
                  <li>Paste both on the right and press <span className="text-fg">Save and test</span>. Strata makes one small request to the stock API and one to the options API to confirm they work.</li>
                </ol>
                <p className="mt-2.5 text-2xs text-muted">The free feed is delayed 15 minutes, which does not matter for history. Requests are throttled to stay inside the free limit of 200 per minute, and unusually large downloads still ask you first.</p>
              </div>
              <div className="flex flex-col gap-1.5">
                {ALPACA_KEYS.map(keyRow)}
                <div className="mt-1.5 flex items-center gap-2">
                  <Button variant="primary" size="sm" disabled={connect.isPending || (!ALPACA_KEYS.every((k) => keys[k] || d.secrets[k]?.set))} onClick={() => connect.mutate()}>{connect.isPending ? <Loader2 className="animate-spin" /> : <PlugZap />} {ALPACA_KEYS.some((k) => keys[k]) ? "Save and test" : "Test connection"}</Button>
                  {!ALPACA_KEYS.every((k) => keys[k] || d.secrets[k]?.set) && <span className="text-2xs text-muted">Both the key ID and the secret are needed.</span>}
                </div>
                {alpaca && (
                  <div className={cn("mt-1 rounded-sm border px-2.5 py-2 text-xs", alpaca.ok ? "border-up/40 bg-up/5" : "border-down/40 bg-down/5")} data-testid="alpaca-result">
                    <div className={alpaca.ok ? "text-up" : "text-down"}>{alpaca.ok ? "Connected" : "Not connected"}</div>
                    <div className="mt-0.5 text-fg-2">{alpaca.message}</div>
                    {alpaca.ok && <div className="mt-1 flex gap-3 text-2xs text-muted"><span>{alpaca.stocks ? "✓" : "✕"} stock bars</span><span>{alpaca.options ? "✓" : "✕"} option contracts and trade bars</span></div>}
                    {alpaca.ok && <div className="mt-1 text-2xs text-muted">Nothing was downloaded. The lab now fetches what a question needs, for the sessions that need it.</div>}
                  </div>
                )}
              </div>
            </div>
          </Panel>
          <Panel title={<span className="flex items-center gap-1.5"><KeyRound className="size-3" /> Model API keys</span>}>
            <div className="flex flex-col gap-1.5 p-3">{Object.values(LLM_KEYS).map(keyRow)}</div>
          </Panel>
          <Panel title={<span className="flex items-center gap-1.5"><KeyRound className="size-3" /> Data provider keys</span>}>
            <div className="flex flex-col gap-1.5 p-3">
              {DATA_KEYS.map(keyRow)}
              <p className="mt-1 text-2xs leading-relaxed text-muted">Yahoo, Cboe, DoltHub and the open option archive need no key. Alpaca's free account adds 1-minute stock bars since 2016 and option trade bars since January 2024. Paid providers are only ever called when you start a download yourself.</p>
            </div>
          </Panel>
          <Panel title="Preferences">
            <div className="grid grid-cols-2 gap-2.5 p-3">
              <Field label="Clock times without a timezone are" hint="“Buy at 10:15” is read in this timezone and converted to New York time. The interpretation panel always shows the converted time."><Select value={draft["ui.timezone"] || "America/Los_Angeles"} onChange={(v) => set("ui.timezone", v)} options={ZONES} /></Field>
              <Field label="Default symbol"><Input className="num uppercase" value={draft["data.default_symbol"] || ""} onChange={(e) => set("data.default_symbol", e.target.value.toUpperCase())} /></Field>
            </div>
          </Panel>
          <Panel title="Model usage (last 30 days)">
            <div className="p-3 text-xs">
              {usage.data?.rows?.length ? (
                <table className="w-full text-2xs">
                  <thead><tr className="text-left text-muted"><th className="pb-1 font-normal">Model</th><th className="pb-1 font-normal">Used for</th><th className="pb-1 text-right font-normal">Calls</th><th className="pb-1 text-right font-normal">Tokens in</th><th className="pb-1 text-right font-normal">Tokens out</th></tr></thead>
                  <tbody>
                    {usage.data.rows.map((r: any, i: number) => <tr key={i} className="border-t border-line"><td className="py-1 font-mono">{r.provider}:{r.model}</td><td className="py-1 text-fg-2">{r.purpose}</td><td className="num py-1 text-right">{r.calls}</td><td className="num py-1 text-right">{(r.input_tokens || 0).toLocaleString()}</td><td className="num py-1 text-right">{(r.output_tokens || 0).toLocaleString()}</td></tr>)}
                    <tr className="border-t border-line text-fg"><td className="py-1" colSpan={2}>Total</td><td className="num py-1 text-right">{usage.data.calls}</td><td className="num py-1 text-right">{usage.data.input_tokens.toLocaleString()}</td><td className="num py-1 text-right">{usage.data.output_tokens.toLocaleString()}</td></tr>
                  </tbody>
                </table>
              ) : <span className="text-muted">No model has been called. The built-in interpreter uses no tokens.</span>}
              <p className="mt-2 text-2xs leading-relaxed text-muted">Token counts are the ones the provider reported for each request. Cost depends on your plan with that provider.</p>
            </div>
          </Panel>
          <Panel title={<span className="flex items-center gap-1.5"><ShieldCheck className="size-3" /> How keys are handled</span>}>
            <ul className="list-inside list-disc space-y-1 p-3 text-2xs leading-relaxed text-muted">
              <li>Keys are stored in the <span className="text-fg-2">{d.secret_store?.label || "local credential store"}</span> on this machine (or read from environment variables) and used only by the backend. They are never written to the database, logs, exports or model prompts.</li>
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
