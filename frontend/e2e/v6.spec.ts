/* End-to-end for V6, through the browser, on the seeded fixture store (backend/tests/e2e_seed.py).
   TST fell 3% or more on two days; only one of the two following sessions has option trade prints. LNG has 1,500 daily
   sessions. Nothing is downloaded, no provider is contacted and no key exists, so every provider-gated state is BLOCKED. */
import { expect, test, type Page } from "@playwright/test";

const ask = async (page: Page, text: string) => {
  const box = page.getByTestId("composer");
  await box.click();
  await box.fill(text);
  await box.press("Enter");
  await expect(page.getByTestId("msg-user").last()).toHaveText(text);
};
const idle = async (page: Page) => {
  await expect(page.getByText("Reading the request…")).toHaveCount(0);
  await expect(page.locator('[data-testid="pipeline"] .animate-spin')).toHaveCount(0, { timeout: 60_000 });
};

test.describe.configure({ mode: "serial" });

test("coverage comes before profitability, in the plan, the result, the report and the comparison", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto("/lab");
  await ask(page, "How often does TST rise 1% after a 3% down day?");
  await expect(page.getByTestId("result-single")).toBeVisible();
  await ask(page, "Test ATM 0DTE calls on those days using historical option trade prices.");
  await idle(page);

  // the plan says how much of the request the stored observations can answer, before anything about profit
  const check = page.getByTestId("data-check").last();
  await expect(check).toHaveAttribute("data-priced", "1");
  await expect(check).toHaveAttribute("data-of", "2");
  await expect(check).toContainText("priced on trade prints");
  await expect(check).toContainText("no option prices stored");

  // the result card leads with the same fraction, above the sentence that holds the profit figure
  const res = page.getByTestId("result-single").last();
  const cov = res.getByTestId("result-coverage");
  await expect(cov).toHaveAttribute("data-priced", "1");
  await expect(cov).toHaveAttribute("data-of", "2");
  const before = await res.evaluate((el) => {
    const c = el.querySelector('[data-testid="result-coverage"]'), t = el.querySelector(".prose");
    return !!(c && t && c.compareDocumentPosition(t) & Node.DOCUMENT_POSITION_FOLLOWING);
  });
  expect(before, "the coverage block comes before the result text").toBe(true);
  // and so does the full report on the right: the fraction sits above the net-result tile
  const report = page.getByTestId("report");
  await expect(report.getByTestId("result-coverage")).toBeVisible();
  const above = await report.evaluate((el) => {
    const c = el.querySelector('[data-testid="result-coverage"]');
    const t = [...el.querySelectorAll("div")].find((d) => d.textContent === "Net result");
    return !!(c && t && c.compareDocumentPosition(t) & Node.DOCUMENT_POSITION_FOLLOWING);
  });
  expect(above, "coverage is above the profit tiles in the report").toBe(true);

  // one workflow for every request: where this one stands, and what each stage opens
  const rail = page.getByTestId("stage-rail");
  for (const s of ["understand", "validate", "plan", "execute", "explain"]) await expect(rail.locator(`[data-stage="${s}"]`)).toHaveAttribute("data-status", "done");
  await expect(rail.locator('[data-stage="data"]')).toHaveAttribute("data-status", "attention");
  await expect(rail.locator('[data-stage="data"]')).toContainText("1/2");
  await rail.locator('[data-stage="data"]').click();
  const view = page.getByTestId("data-check-view");
  await expect(view.getByTestId("data-check")).toHaveAttribute("data-of", "2");
  await expect(view.getByTestId("session-list").locator("tbody tr")).toHaveCount(1); // the one session that cannot be tested
  await expect(view.getByTestId("session-list")).toContainText("no option prices stored");
  await expect(view.getByTestId("alternatives")).toContainText("Run the underlying market study");
  await expect(view.getByTestId("alternatives").locator('[data-alt="modeled"]')).toContainText("opt in");

  // three questions about the observations themselves
  await ask(page, "Show every actual option fill and its source.");
  await expect(page.getByTestId("result-fills")).toContainText("1 option fill in 1 trade");
  const fills = page.getByTestId("fills-view");
  await expect(fills).toContainText("trade print");
  await expect(fills).toContainText("none observed"); // no bid/ask is shown for a print
  await expect(fills).toContainText("Nothing here is modelled");
  await ask(page, "Show exactly which historical sessions could not be tested.");
  await expect(page.getByTestId("result-untested")).toContainText("1 of the 2 qualifying sessions did not pass every check on stored observations");
  await expect(page.getByTestId("untested-list").locator("tbody tr")).toHaveCount(1);
  await ask(page, "Estimate what it would cost to acquire the missing observations.");
  await expect(page.getByTestId("result-cost")).toContainText("priced by what is stored now");
  await expect(page.getByTestId("cost-view").getByTestId("budget-table")).toContainText("Strata never buys anything");
  await expect(rail.locator('[data-stage="followup"]')).toHaveAttribute("data-status", "done");

  // a comparison of variants states each one's coverage first, in the sentence and in the table
  await ask(page, "Compare 0DTE and 7DTE calls.");
  await idle(page);
  const sweep = page.getByTestId("result-sweep").last();
  // the fixture stores same-day contracts only. Since V7 a 7DTE leg is never filled with the 0DTE contract that happens to be
  // stored, so that variant does not run, and the comparison says so instead of comparing a strategy with a substitute for it
  await expect(sweep).toContainText("Only one of the 2 variants could be run");
  await expect(sweep).toContainText("not an expiration the strategy can use");
  await expect(sweep.locator("thead")).toContainText("Rule fired");
  await expect(sweep.locator("thead")).toContainText("Priced");
  expect(errors).toEqual([]);
});

test("data readiness is counted from the stored files, and a blocked provider says exactly what it needs", async ({ page, request }) => {
  await page.goto("/data");
  const r = page.getByTestId("readiness-page");
  await expect(r).toContainText("What this data can answer today");
  await r.getByTestId("readiness-TST").click();
  const tbl = r.getByTestId("capability-table");
  await expect(tbl.locator('[data-capability="daily"]')).toHaveAttribute("data-ready", "yes");
  await expect(tbl.locator('[data-capability="option_trades"]')).toHaveAttribute("data-ready", "yes");
  await expect(tbl.locator('[data-capability="option_quotes"]')).toHaveAttribute("data-ready", "no");
  await r.getByTestId("readiness-LNG").click();
  await expect(tbl.locator('[data-capability="intraday"]')).toHaveAttribute("data-ready", "no");

  // no key exists here: the state is BLOCKED, never "working", and the steps name the page the key is typed on
  const acc = r.getByTestId("alpaca-access");
  await expect(acc).toHaveAttribute("data-status", "blocked");
  await expect(acc).toContainText("BLOCKED, no credentials");
  const how = acc.getByTestId("alpaca-how");
  await expect(how.locator("li")).toHaveCount(4);
  await expect(how).toContainText("Connect a free Alpaca account");
  await expect(how).toContainText("never written to a log, an export or a language-model prompt");
  await expect(acc.getByTestId("enter-key")).toHaveAttribute("href", "/settings#alpaca");
  const rep = await (await request.get("/api/data/access/alpaca")).json();
  expect(rep.verified).toBe(false);
  expect(rep.status).toBe("blocked");
  expect(JSON.stringify(rep).toLowerCase()).not.toContain("secret\":\"");

  // the model-backed reader is reported as not run, with the way to configure it, never as passing
  await page.goto("/settings");
  const ev = page.getByTestId("eval-panel");
  await expect(ev.getByTestId("eval-llm")).toContainText("Model-backed reading: NOT RUN");
  await expect(ev.getByTestId("eval-llm")).toContainText("Model API keys");
  await expect(ev.getByRole("button", { name: "Measure with the configured model" })).toBeDisabled();
  await expect(ev.getByTestId("eval-run")).toBeEnabled();

  // the research console reads the store and nothing else
  const ok = await request.post("/api/research/sql", { data: { sql: "SELECT provider, symbol, count(*) AS n FROM daily GROUP BY ALL ORDER BY symbol" } });
  expect(ok.status()).toBe(200);
  expect((await ok.json()).rows.map((x: any[]) => x[1])).toContain("TST");
  for (const sql of ["SELECT * FROM read_text('/etc/hosts')", "SELECT 1; SELECT 2", "COPY (SELECT 1) TO '/tmp/strata-e2e-should-not-exist.csv'", "SELECT * FROM 'https://example.invalid/x.csv'"]) {
    const bad = await request.post("/api/research/sql", { data: { sql } });
    expect(bad.status(), sql).toBe(400);
  }
});

test("a frozen experiment is dated by the clock, counts from the next session, and keeps a sealed ledger", async ({ page, request }) => {
  await page.goto("/forward");
  const runs = await (await request.get("/api/backtests?kind=backtest,graph&limit=20")).json(); // what the page offers to freeze
  const run = runs.find((x: any) => x.status === "done" && x.summary?.outcome !== "needs_decision"); // a run the data could not cover has nothing to freeze, and the page does not offer it
  expect(run).toBeTruthy();
  await page.getByTestId("forward").locator("select").selectOption(run.id);
  await page.getByTestId("forward-freeze").click();
  const d = page.getByTestId("forward-detail");
  await expect(d.getByTestId("forward-activated")).toContainText(/activated \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC/);
  await expect(d.getByTestId("forward-first")).toContainText(/first session that counts \d{4}-\d{2}-\d{2}/);
  await expect(d.getByTestId("forward-ledger")).toHaveAttribute("data-ok", "yes");
  await expect(d.getByTestId("forward-ledger")).toContainText("chain verified");
  const ex = (await (await request.get("/api/forward")).json()).experiments.find((e: any) => e.source_run_id === run.id);
  // the activation instant is the clock's; the first session that counts is strictly after it, whatever is stored
  expect(new Date(ex.activated_at).getTime()).toBeGreaterThan(Date.now() - 5 * 60_000);
  // (sessions are New York dates: compare with the activation's New York date, not its UTC date, which is the next day every evening)
  const activatedNy = new Date(ex.activated_at).toLocaleDateString("en-CA", { timeZone: "America/New_York" });
  expect(ex.first_session > activatedNy || ex.activation.state === "before_open").toBe(true);
  await d.getByTestId("forward-evaluate").click();
  await expect(d).toContainText("No session since activation has closed yet", { timeout: 30_000 });
  const led = await (await request.get(`/api/forward/${ex.id}/ledger`)).json();
  expect(led.verification.ok).toBe(true);
  // the ledger opens with the activation itself; nothing is recorded for a session that has not happened
  expect(led.entries.map((e: any) => e.kind)).toEqual(["activation"]);
  expect(led.entries[0].body.config_hash).toBe(ex.config_hash);
});

test("an ML study is validated first; the final test set is scored once, for the frozen model, and the look is recorded", async ({ page, request }) => {
  test.setTimeout(300_000);
  await page.goto("/ml");
  await page.getByLabel("Symbol").selectOption("LNG");
  await page.getByTestId("ml-run").click();
  const res = page.getByTestId("ml-result");
  await expect(res).toBeVisible({ timeout: 150_000 });
  const ft = res.getByTestId("ml-final-test");
  await expect(ft).toHaveAttribute("data-scored", "no");
  await expect(ft).toContainText("Final test set: not looked at");
  await expect(res).toContainText("only the selected model is ever scored on the final test set");
  await expect(res).toContainText("not scored"); // the tile, where a test number used to be
  const first = page.url().split("/ml/")[1];
  const v = await (await request.get(`/api/runs/${first}/ml`)).json();
  expect(v.final_test).toBe(false);
  expect(v.models.every((m: any) => m.test === null || m.test === undefined)).toBe(true);
  expect(v.test_exposure.scored).toBe(false);

  if (v.verdict === "leak_suspected") {
    await expect(ft.getByTestId("ml-score-final")).toBeDisabled();
    return;
  }
  await ft.getByTestId("ml-score-final").click();
  await expect(ft.getByTestId("ml-final-confirm")).toContainText("This is the one look at the final test set");
  await ft.getByTestId("ml-final-go").click();
  await expect(page).not.toHaveURL(new RegExp(`/ml/${first}$`), { timeout: 150_000 });
  const scored = page.getByTestId("ml-final-test");
  await expect(scored).toHaveAttribute("data-scored", "yes");
  const second = page.url().split("/ml/")[1];
  const f = await (await request.get(`/api/runs/${second}/ml`)).json();
  expect(f.final_test).toBe(true);
  expect(f.validation_run).toBe(first);
  expect(f.frozen.config_key).toBe(v.frozen.config_key);
  expect(f.models.filter((m: any) => m.test).length).toBe(1);
  expect(f.models.find((m: any) => m.test).name).toBe(v.selected);
  expect(["confirmed_once", "not_confirmed"]).toContain(f.confirmation);
  await expect(page.getByTestId("ml-confirmation")).toHaveAttribute("data-confirmation", f.confirmation);
  // the validation run now shows that its test period has been looked at, and offers no way to undo it
  await page.goto(`/ml/${first}`);
  await expect(page.getByTestId("ml-final-test")).toHaveAttribute("data-scored", "no");
  const again = await request.post(`/api/runs/${second}/ml/final-test`);
  expect(again.status()).toBe(409);
});

test("a copy on standby says so on the whole page and continues by itself", async ({ page }) => {
  // Another copy of Strata (across a container boundary) holds the data folder: this copy has not opened the database.
  let standby = true;
  const holder = { where: "the app running directly on this Mac", pid: 4242, role: "server", url: "http://localhost:8765", engine_version: "6.0.0", started: "2026-11-26T06:03:10+00:00" };
  await page.route("**/api/health", async (route) => {
    if (!standby) return route.continue();
    await route.fulfill({ json: { ok: true, engine_version: "6.0.0", data_dir: "/app/data", standby: { standby: true, holder, checking: false, message: "This data directory is in use by another copy of Strata" } } });
  });
  await page.route("**/api/**", async (route) => {
    if (!standby || route.request().url().endsWith("/api/health")) return route.fallback();
    await route.fulfill({ status: 503, json: { detail: "This data directory is in use by another copy of Strata", standby: { standby: true, holder } } });
  });
  await page.goto("/settings");
  const sb = page.getByTestId("standby");
  await expect(sb).toBeVisible();
  await expect(sb).toContainText("This copy of Strata is not running the app");
  await expect(sb.getByTestId("standby-holder")).toHaveText("the app running directly on this Mac");
  await expect(sb).toContainText("engine 6.0.0");
  await expect(sb).toContainText("nothing entered here would be saved");
  await expect(sb.getByTestId("standby-open")).toHaveAttribute("href", "http://localhost:8765");
  // nothing of the application is on the page: no navigation, no form that could take a key
  await expect(page.locator("#alpaca")).toHaveCount(0);
  await expect(page.locator("input[type=password]")).toHaveCount(0);
  // the other copy stops: the page carries on without a reload
  standby = false;
  await expect(sb).toHaveCount(0, { timeout: 15_000 });
  await expect(page.locator("#alpaca")).toBeVisible({ timeout: 15_000 });
});

test("a key stored where this copy cannot read it is accounted for, not reported as missing", async ({ page }) => {
  // inside a container: the key was saved in the macOS Keychain by the app running directly on the Mac
  await page.route("**/api/settings", async (route) => {
    if (route.request().method() !== "GET") return route.continue();
    const res = await route.fetch();
    const body = await res.json();
    for (const k of ["ALPACA_API_KEY", "ALPACA_API_SECRET"]) {
      body.secrets[k] = { ...body.secrets[k], set: false, source: null, hint: null, elsewhere: { label: "macOS Keychain", from: "the app running directly on this Mac", at: "2026-11-26T05:14:00+00:00",
        note: "A key was saved in the macOS Keychain by the app running directly on this Mac. This copy cannot read that store: enter the key here as well to use it from here." } };
    }
    await route.fulfill({ json: body });
  });
  await page.goto("/settings#alpaca");
  const note = page.getByTestId("elsewhere-ALPACA_API_KEY");
  await expect(note).toBeVisible();
  await expect(note).toContainText("saved in the macOS Keychain");
  await expect(note).toContainText("enter the key here as well");
  await expect(page.getByTestId("elsewhere-ALPACA_API_SECRET")).toBeVisible();
  await expect(page.locator("body")).not.toContainText("PK"); // where the key is, never the key
});
