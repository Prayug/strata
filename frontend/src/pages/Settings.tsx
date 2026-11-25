import { useMutation, useQuery } from "@tanstack/react-query";
import { Check, ExternalLink, KeyRound, Loader2, PlugZap, ShieldCheck, X } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { ErrorBox, Loading, PageHeader, Panel } from "@/components/common";
import { Button, Field, Input, Select } from "@/components/ui";
import { api, queryClient } from "@/lib/api";
import { cn } from "@/lib/format";

const LLM_KEYS: Record<string, string> = { anthropic: "ANTHROPIC_API_KEY", openai: "OPENAI_API_KEY", gemini: "GEMINI_API_KEY", openrouter: "OPENROUTER_API_KEY" };
const DATA_KEYS = ["POLYGON_API_KEY", "DATABENTO_API_KEY", "ORATS_API_KEY"];
const ALPACA_KEYS = ["ALPACA_API_KEY", "ALPACA_API_SECRET"];
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
      <div key={name} className="flex items-center gap-2">
        <span className="w-36 shrink-0 text-xs text-fg-2">{s.label}</span>
        <Input sz="sm" type="password" autoComplete="off" className="num flex-1" placeholder={s.set ? `${s.hint || "set"}  (${s.source})` : "not set"} value={keys[name] || ""} onChange={(e) => setKeys((k) => ({ ...k, [name]: e.target.value }))} />
        {s.set && s.source !== "environment variable" && <Button size="iconSm" variant="ghost" title="Remove stored key" onClick={() => clearKey(name)}><X /></Button>}
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
                <div className="grid grid-cols-2 gap-2.5">
                  <Field label="If the model fails or is unavailable" hint="Either way you are told which interpreter read each request; a complex strategy is never silently handed to a weaker one.">
                    <Select value={draft["llm.fallback"] || "rules"} onChange={(v) => set("llm.fallback", v)} options={[{ value: "rules", label: "Use the built-in interpreter and say so" }, { value: "none", label: "Stop and report the failure" }]} />
                  </Field>
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
          <Panel title={<span className="flex items-center gap-1.5"><PlugZap className="size-3" /> Connect a free Alpaca account</span>} className="col-span-2">
            <div id="alpaca" className="grid grid-cols-[1.1fr_1fr] gap-5 p-3">
              <div className="text-xs leading-relaxed text-fg-2">
                <p className="text-muted">Alpaca's free account is what makes intraday option backtests possible without paying for data: 1-minute stock bars since 2016 and 1-minute option trade bars since February 2024. Strata only reads market data with it; it never places an order.</p>
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
              <p className="mt-1 text-2xs leading-relaxed text-muted">Yahoo, Cboe, DoltHub and the open option archive need no key. Alpaca's free account adds 1-minute bars since 2016. Paid providers are only ever called when you start a download yourself.</p>
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
