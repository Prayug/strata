/* End-to-end for the V4 research tools, through the browser, on the seeded fixture store.
   LNG (backend/tests/e2e_seed.py) has 1,500 daily sessions in which a fall of 2% or more is followed by a session
   that closes about 1.2% above its open: a real, directional effect the plan must find, and one the reviewer must
   still question. Nothing is downloaded. */
import { expect, test, type Page } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

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
let notebook = "";

test("a broad question becomes a bounded plan, every configuration is shown, and a candidate is taken further", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto("/");
  await ask(page, "Investigate whether LNG tends to reverse following large down days.");
  await expect(page).toHaveURL(/\/lab\/[0-9a-f]+/);
  notebook = page.url();
  // the plan is stated before anything is computed: hypothesis, null, bounds, data
  const plan = page.getByTestId("plan-card");
  await expect(plan).toContainText("After a large down day, LNG tends to reverse");
  await expect(plan).toContainText("of at most 60");
  await expect(plan).toContainText("Minimum sample");
  await expect(plan).toContainText("final test (scored once, for candidates only)");
  await expect(plan).toContainText("Daily bars LNG: 1,500 sessions");
  // the result: all 36 configurations, including the ones that showed nothing
  const res = page.getByTestId("result-hypothesis");
  await expect(res).toBeVisible({ timeout: 60_000 });
  await expect(res).toContainText("36 configurations were evaluated");
  await expect(res).toContainText("separate test on observed premiums");
  const view = page.getByTestId("hypothesis-view");
  await expect(view).toBeVisible();
  await expect(view.getByText("All 36 configurations")).toBeVisible();
  const rows = view.locator('[data-testid="hypothesis-table"] tbody tr[id^="cfg-"]');
  await expect(rows).toHaveCount(36);
  await expect(view.getByText("Confirmed on untouched data").first()).toBeVisible();
  await expect(view.getByText("Did not qualify").first()).toBeVisible();
  // the final test period was scored once and the notebook says so
  await expect(page.getByTestId("holdout-chip").first()).toContainText("used once");
  await expect(page.getByTestId("trials-chip")).toContainText("36 tested");
  // a configuration that was NOT a candidate was never scored on the final period
  await rows.filter({ hasText: "Did not qualify" }).first().click();
  await expect(view.getByText("Final test: not scored")).toBeVisible();
  await view.getByTestId("see-sessions").click();
  await expect(page.getByRole("dialog")).toContainText("Sessions of the final test period are not listed");
  await page.keyboard.press("Escape");
  // take the first candidate further: the same configuration, session by session
  await res.getByRole("button", { name: "Investigate further" }).first().click();
  await idle(page);
  const single = page.getByTestId("result-single").last();
  await expect(single).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId("msg-user").last()).toContainText("Investigate further:");
  await expect(page.getByTestId("plan-note").last()).toContainText("not counted as a new hypothesis");
  // the same configuration run session by session is not a new hypothesis: the count stays at 36 and the final period is not contaminated
  await expect(page.getByTestId("trials-chip")).toContainText("36 tested");
  await expect(page.getByTestId("holdout-chip").first()).toContainText("used once");
  expect(errors).toEqual([]);
});

test("the reviewer examines all sixteen topics, runs its tests, and a reasoning question is answered from the observations", async ({ page }) => {
  await page.goto(notebook);
  await ask(page, "Challenge this result");
  const card = page.getByTestId("result-challenge");
  await expect(card).toBeVisible({ timeout: 90_000 });
  const view = page.getByTestId("challenge-view");
  await expect(view).toBeVisible();
  await expect(view.locator('[data-testid^="finding-"]')).toHaveCount(16);
  const look = view.getByTestId("finding-lookahead");
  await expect(look).toHaveAttribute("data-severity", "ok");
  await expect(look).toContainText("future-scramble test");
  await expect(look).toContainText("scrambled");
  await expect(view.getByTestId("finding-selection")).toContainText("36 configurations");
  await expect(view.getByTestId("finding-fills")).toHaveAttribute("data-severity", "na");
  await expect(view.getByText("Tests run:")).toBeVisible();
  // every severity is a glyph and a word, never a colour alone
  await expect(view.getByText(/Checked, nothing found|Weakness|Serious|Minor/).first()).toBeVisible();
  await ask(page, "How much of the result is explained by volatility?");
  const ans = page.getByTestId("result-analysis").last();
  await expect(ans).toBeVisible();
  await expect(page.getByTestId("analysis-view")).toContainText("volatility");
  await expect(page.getByTestId("analysis-view")).toContainText("calculated from the stored observations");
});

test("the canvas shows the notebook as a graph, opens real results, branches, and keeps its layout", async ({ page }) => {
  await page.goto(notebook);
  await page.getByTestId("open-canvas").click();
  const canvas = page.getByTestId("canvas");
  await expect(canvas).toBeVisible();
  for (const t of ["question", "plan", "tested", "candidate", "study", "challenge", "analysis"]) await expect(canvas.getByTestId(`node-${t}`).first()).toBeVisible();
  const nodes = ".react-flow__node";
  const before = await canvas.locator(nodes).count();
  expect(before).toBeGreaterThanOrEqual(9);
  // collapsing the group of tested configurations hides its candidates and what hangs below them
  await canvas.getByTestId("node-tested").getByTestId("node-collapse").click();
  await expect(canvas.getByTestId("node-candidate")).toHaveCount(0);
  await expect(canvas.getByTestId("node-tested")).toContainText(/Show \d+ hidden/);
  await page.waitForTimeout(900); // the layout is saved a moment after the change
  await page.reload();
  await page.getByTestId("open-canvas").click();
  await expect(page.getByTestId("canvas").getByTestId("node-tested")).toContainText(/Show \d+ hidden/); // persisted with the notebook
  await page.getByTestId("canvas").getByTestId("node-tested").getByTestId("node-collapse").click();
  await expect(page.getByTestId("canvas").locator(nodes)).toHaveCount(before);
  // a node carries its real configuration and can be branched from
  await page.getByTestId("canvas").getByTestId("node-study").first().click();
  const detail = page.getByTestId("node-detail");
  await expect(detail).toContainText("Configuration");
  await expect(detail).toContainText("Branch from here");
  await detail.getByRole("button", { name: "Are most profits", exact: false }).or(detail.getByRole("button", { name: "Test whether the results hold up out of sample" })).first().click();
  // the branch appears on the canvas as new nodes connected to the study; the canvas stays in view
  await expect.poll(async () => page.getByTestId("canvas").locator(nodes).count(), { timeout: 60_000 }).toBeGreaterThan(before);
  await expect(page.getByTestId("canvas").locator(".react-flow__edge").first()).toBeAttached();
  // the selected study node opens its real report
  await page.getByTestId("node-detail").getByTestId("node-open").click();
  await expect(page.getByTestId("canvas")).toHaveCount(0);
  await expect(page.getByText("On all sessions").first()).toBeVisible();
});

test("the sensitivity map marks thin and stable cells and every cell opens its sessions", async ({ page }) => {
  const t0 = Date.now();
  await page.goto("/sensitivity");
  const map = page.getByTestId("heatmap");
  await expect(map).toBeVisible();
  const cells = map.getByTestId("cell");
  await expect(cells).toHaveCount(35);
  const rendered = Date.now() - t0;
  await expect(map).toContainText("fewer than 30 observations");
  await expect(map).toContainText("stable region");
  expect(await map.locator('[data-testid="cell"][data-stable="1"]').count()).toBeGreaterThan(0);
  await page.getByTestId("layer-probability").click();
  await expect(cells.first()).toContainText("%");
  await cells.nth(3).click();
  const panel = page.getByTestId("cell-panel");
  await expect(panel).toContainText("Observations");
  await expect(panel).toContainText("You are choosing one of 35 cells");
  await expect(panel.locator("table tbody tr").first()).toBeVisible();
  await panel.getByTestId("cell-to-lab").click();
  await expect(page).toHaveURL(/\/lab\/[0-9a-f]+/);
  await expect(page.getByTestId("result-single")).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId("plan-note").last()).toContainText("picked from a map of 35 cells");
  await expect(page.getByTestId("trials-chip")).toContainText("35 tested");
  // a persistent record of how long the map took to appear, for regressions across commits
  const out = path.resolve(here, "../../benchmarks");
  try {
    fs.mkdirSync(out, { recursive: true });
    fs.appendFileSync(path.join(out, "frontend.jsonl"), JSON.stringify({ at: new Date().toISOString(), workload: "sensitivity_map_35_cells_ms", ms: rendered, ci: !!process.env.CI }) + "\n");
  } catch { /* recording is best effort */ }
  expect(rendered).toBeLessThan(15_000);
});

test("an encyclopedia entry is computed from stored data, reconfigures, and becomes a hypothesis", async ({ page }) => {
  await page.goto("/encyclopedia/decline-recovery");
  const entry = page.getByTestId("ency-entry");
  await expect(entry.getByRole("heading", { name: "Recovery after a large decline" })).toBeVisible();
  const summary = page.getByTestId("ency-summary");
  await expect(summary).toContainText("LNG fell 2% or more on");
  const first = await summary.innerText();
  await entry.getByLabel("Decline (%)").fill("3");
  await page.getByTestId("ency-apply").click();
  await expect(summary).toContainText("LNG fell 3% or more on");
  expect(await summary.innerText()).not.toEqual(first); // recomputed, not a stored sentence
  await expect(page.getByTestId("ency-table").locator("tbody tr")).toHaveCount(12);
  await page.getByTestId("ency-table").locator("tbody tr").first().click();
  await expect(page.getByTestId("cell-panel")).toContainText("Observations");
  await page.getByTestId("to-hypothesis").click();
  await expect(page).toHaveURL(/\/lab\/[0-9a-f]+/);
  await expect(page.getByTestId("plan-card")).toBeVisible({ timeout: 30_000 });
});

test("the collector, source verification and bulk import are operated from the data page", async ({ page }) => {
  await page.goto("/data");
  await page.getByRole("tab", { name: /Collector/ }).click();
  const col = page.getByTestId("collector");
  await expect(col).toContainText("It only reads market data");
  await expect(col).toContainText("cannot place, change or read orders");
  await expect(col.getByTestId("coverage-calendar")).toBeVisible();
  await expect(col.getByTestId("collector-enabled")).not.toBeChecked(); // off until switched on
  await col.getByLabel("Strikes each side").fill("6");
  await col.getByTestId("collector-save").click();
  await expect(page.getByText("Collector settings saved")).toBeVisible();
  await page.reload();
  await page.getByRole("tab", { name: /Collector/ }).click();
  await expect(page.getByTestId("collector").getByLabel("Strikes each side")).toHaveValue("6");
  // a source with no credentials is unverified, whatever its tests say
  await page.getByTestId("tab-sources").click();
  const src = page.getByTestId("marketplace");
  await expect(src.getByTestId("source-alpaca").getByTestId("verification")).toContainText("unverified");
  await expect(src).toContainText("Files you obtain and import");
  // bulk import (an advanced tool): the layout is recognised from the columns, then streamed in
  await page.getByRole("button", { name: /advanced tools/ }).click();
  await page.getByRole("tab", { name: /Bulk import/ }).click();
  const imp = page.getByTestId("bulk-import");
  await imp.getByTestId("import-path").fill(path.join(process.env.E2E_DATA_DIR || "", "fixtures", "vendor_quotes.zip"));
  await imp.getByLabel("Dataset name").fill("vendorfile");
  await imp.getByTestId("import-inspect").click();
  await expect(imp).toContainText("1 importable file");
  await expect(imp).toContainText("historicaloptiondata.com option quotes");
  await imp.getByTestId("import-go").click();
  await expect(page.getByText("Data job finished")).toBeVisible({ timeout: 60_000 });
  await page.getByRole("tab", { name: /Datasets/ }).click();
  await expect(page.getByText("vendorfile").first()).toBeVisible();
});
