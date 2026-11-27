/* End-to-end for V7, through the browser, on the seeded fixture store (backend/tests/e2e_seed.py).
   TST fell 3% or more on two days; the session after the first of them has option trade prints for the 96 call, the other
   has none. No provider is contacted and no key exists: the individual-trade check and the quote comparison are BLOCKED
   here, and the test asserts that they say so instead of showing a number. */
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
let notebook = "";

test("the desk: a sentence on the command line, its reading as a ticket, and the result with its trades, replay and fills on one screen", async ({ page, request }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto("/lab");
  await ask(page, "Buy ATM TST 0DTE calls at the open after a 3% down day, sell at 3:55 PM Eastern.");
  await expect(page).toHaveURL(/\/lab\/[0-9a-f]+/);
  notebook = page.url();
  await idle(page);
  // the command line stays on top, and the reading of the sentence is a ticket under it
  await expect(page.getByTestId("desk")).toBeVisible();
  const ticket = page.getByTestId("ticket");
  await expect(ticket).toContainText("TST");
  await expect(ticket).toContainText("previous session fell 3% or more");
  await expect(ticket).toContainText("Long call");
  await expect(page.getByTestId("stage-rail")).toBeVisible();
  await expect(page.getByTestId("result-single").last()).toBeVisible({ timeout: 60_000 });
  // the tear sheet: the sample and the coverage first, and no claim of an edge from one trade
  await page.getByTestId("tab-performance").click();
  const perf = page.getByTestId("view-performance");
  await expect(perf.getByTestId("perf-evidence")).toHaveText(/too few trades to judge/i);
  await expect(perf.getByTestId("perf-trades")).toContainText("1 trades on 1 sessions");
  await expect(perf.getByTestId("perf-coverage")).toContainText("the entry rule fired on 2 sessions");
  await expect(perf.getByTestId("perf-coverage")).toContainText("1 of them (50.0%) could be priced");
  await expect(perf.getByTestId("perf-baseline")).toContainText("Not run yet");
  // every fill, against the stored bar it was priced from
  await page.getByTestId("tab-fills").click();
  const fills = page.getByTestId("view-fills");
  await expect(fills).toContainText("2 of 2");
  await expect(fills.getByTestId("fills-table").locator("tbody tr")).toHaveCount(2);
  const csv = await (await request.get((await fills.getByTestId("fills-csv").getAttribute("href"))!)).text();
  expect(csv.split("\n")[0]).toContain("matches_stored_bar");
  expect(csv.split("\n").filter(Boolean).length).toBe(3);
  // a fill opens the replay of its trade: the stored bars of the underlying and of the contract, and nothing between prints
  await fills.getByTestId("fills-table").locator("tbody tr").first().click();
  const replay = page.getByTestId("view-replay");
  await expect(replay).toBeVisible();
  await expect(replay.getByTestId("replay-contract-bar")).toContainText(/6 of 390 minutes have a print/);
  await expect(replay).toContainText("trade prints (no bid/ask)");
  await expect(replay.getByTestId("replay-readout")).toContainText("position");
  await replay.getByTestId("replay-slider").fill("70");
  await expect(replay.getByTestId("replay-readout")).toContainText("10:41 ET");
  await expect(replay.getByTestId("replay-readout")).toContainText(/last print 2\.00 at 10:31 \(9 min old\)/);
  await replay.getByTestId("replay-known").click();
  await expect(replay.getByTestId("replay-known")).toHaveClass(/on/);
  // execution: without a provider key the comparison on individual prints is blocked, and so are quotes. No number stands in.
  await page.getByTestId("tab-execution").click();
  const exe = page.getByTestId("view-execution-check");
  await expect(exe.getByTestId("quotes-blocked")).toContainText("BLOCKED");
  await expect(exe.getByTestId("hires-run")).toBeDisabled();
  await expect(exe.getByTestId("hires-tick")).toHaveText("–");
  expect(errors).toEqual([]);
});

test("a comparison typed on the same line runs as one grid, and every cell counts as a trial", async ({ page, request }) => {
  await page.goto(notebook);
  await ask(page, "Compare exits after 15, 60 and 120 minutes.");
  await expect(page.getByTestId("pending-experiment")).toContainText("3 configurations");
  await expect(page.getByTestId("pending-experiment")).toContainText("Every configuration counts as a trial");
  const res = page.getByTestId("result-experiment");
  await expect(res).toBeVisible({ timeout: 60_000 });
  const view = page.getByTestId("view-experiment");
  await expect(view).toBeVisible();
  await expect(view.getByTestId("experiment-summary")).toContainText("3 of 3 configurations traded");
  await expect(view.getByTestId("experiment-summary")).toContainText("neither is evidence of an edge");
  await expect(view.getByTestId("experiment-table").locator("tbody tr")).toHaveCount(3);
  await expect(page.getByTestId("trials-chip")).toContainText("4 tested");
  // a cell opens its trades, down to the contract and the prices
  await view.getByTestId("experiment-table").locator("tbody tr").first().click();
  await expect(view.getByTestId("experiment-cell")).toContainText("96C");
  const csv = await (await request.get((await view.getByTestId("experiment-csv").getAttribute("href"))!)).text();
  expect(csv.split("\n")[0]).toContain("setting_exit_minutes");
  expect(csv.split("\n").filter(Boolean).length).toBe(4);
  // a comparison with nothing to apply it to asks which strategy, and runs nothing
  await page.goto("/lab");
  await ask(page, "Compare exits after 15, 30 and 60 minutes.");
  await expect(page.getByTestId("conversation")).toContainText("Which strategy should these be compared on?");
});

test("stored sessions can be replayed without a run, and the collector says which source it continues", async ({ page }) => {
  await page.goto("/replay");
  await expect(page.getByTestId("replay-start")).toContainText("stored observations only");
  await page.goto("/data?tab=collector");
  const col = page.getByTestId("collector");
  await expect(col).toBeVisible();
  await expect(col.getByTestId("collector-sources")).toContainText("Option source");
  await page.goto("/data?tab=readiness");
  const dbn = page.getByTestId("databento-access");
  await expect(dbn).toContainText(/blocked: no key/i);
  await expect(dbn).toContainText("https://databento.com");
  await expect(dbn.getByTestId("databento-sample")).toHaveCount(0); // nothing billable can be asked for without a key
});

test("a question about the first candle and the day is two directions, never a study of nothing; a correction is applied or asked about", async ({ page }) => {
  // typed into the running application by its user, three times in different words: each time it was answered with the share
  // of ALL sessions that closed higher. The fixture's first candles all close at the open, so the honest answer here is
  // that there is no direction to compare: the test is that this is said, and that no frequency appears in its place
  await page.goto("/lab");
  await ask(page, "How often does TST's first 9:30 candle predict its overall day?");
  await expect(page.getByTestId("pending-investigation")).toContainText("the direction of the first 5/15/30 minutes");
  await expect(page.getByTestId("pending-investigation")).toContainText("was not stated, so 5-, 15- and 30-minute candles are all measured");
  const res = page.getByTestId("result-investigation");
  await expect(res).toBeVisible({ timeout: 90_000 });
  await expect(res).toContainText("the first candle closed exactly at the session's open every time");
  await expect(res).not.toContainText(/ended higher|met the condition/);
  await expect(page.getByTestId("result-single")).toHaveCount(0);
  await expect(page.getByTestId("view-investigation")).toContainText("no direction to compare");
  await expect(page.getByTestId("agreement")).toHaveCount(0); // nothing to draw, so nothing is drawn
  // a frequency question with nothing to count is not run as a study either
  await page.goto("/lab");
  await ask(page, "How often does TST do the thing with the wick?");
  await expect(page.getByTestId("conversation")).toContainText("I could not find anything to count in that question");
  await expect(page.getByTestId("result-single")).toHaveCount(0);
  // a correction that is a value on its own: asked about when it fits two things, applied when it fits one
  await page.goto("/lab");
  await ask(page, "How often does TST rise 1% from the open after a 3% down day?");
  await idle(page);
  await expect(page.getByTestId("result-single")).toBeVisible({ timeout: 60_000 });
  await ask(page, "no I meant 2%");
  await expect(page.getByTestId("conversation")).toContainText('"2%" could replace more than one thing in the current reading');
  await expect(page.getByTestId("conversation")).toContainText("Nothing was changed");
  await ask(page, "no I meant 6:45a, PST");
  await expect(page.getByTestId("ticket")).toContainText("09:45 ET");
  await expect(page.getByTestId("ticket").locator('[title*="6:45a, PST"]')).toHaveCount(1); // the words it was read from stay with the value
});
