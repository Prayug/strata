/* End-to-end for V5, through the browser, on the seeded fixture store (backend/tests/e2e_seed.py).
   LNG has 1,500 daily sessions in which a fall of 2% or more is followed by a stronger session; TST has eight
   sessions of one-minute bars. Nothing is downloaded and no provider is contacted. */
import { expect, test, type Page } from "@playwright/test";

const ask = async (page: Page, text: string) => {
  const box = page.getByTestId("composer");
  await box.click();
  await box.fill(text);
  await box.press("Enter");
  await expect(page.getByTestId("msg-user").last()).toHaveText(text);
};

test.describe.configure({ mode: "serial" });
let notebook = "";

test("the application is organised around Explore, Build and Validate, with a simple and an advanced level", async ({ page }) => {
  await page.goto("/lab");
  for (const g of ["explore", "build", "validate"]) await expect(page.getByTestId(`nav-${g}`)).toBeVisible();
  await expect(page.getByTestId("nav-explore")).toContainText("Ask");
  await expect(page.getByTestId("nav-build")).toContainText("Strategy IDE");
  await expect(page.getByTestId("nav-validate")).toContainText("Forward validation");
  // simple mode keeps the specialist tools out of the way; advanced adds them; neither changes any calculation
  await expect(page.getByTestId("nav-explore")).not.toContainText("ML research");
  await expect(page.getByTestId("start-points")).toHaveCount(0);
  await page.getByTestId("mode-toggle").getByRole("button", { name: "advanced" }).click();
  await expect(page.getByTestId("nav-explore")).toContainText("ML research");
  await expect(page.getByTestId("nav-validate")).toContainText("Optimization");
  await expect(page.getByTestId("start-points")).toBeVisible();
  await page.getByTestId("mode-toggle").getByRole("button", { name: "simple" }).click();
  // a first run is told what it can do with what is stored, and what more data would add
  const ob = page.getByTestId("onboarding");
  await expect(ob).toContainText("Free price history");
  await expect(ob).toContainText("Option prices for backtests");
  await expect(page.getByTestId("workflows")).toContainText("Find out whether it survives contact with real prices.");
  await expect(page.getByTestId("unlock")).toContainText("unverified");
  await ob.getByTestId("onboarding-dismiss").click();
  await expect(page.getByTestId("onboarding")).toHaveCount(0);
});

test("a composed question becomes a typed research graph with a graded finding", async ({ page }) => {
  await page.goto("/lab");
  await ask(page, "Does LNG tend to recover after down days of 2% or more, even after controlling for volatility?");
  await page.waitForURL(/\/lab\//);
  notebook = page.url().split("/lab/")[1];
  const card = page.getByTestId("graph-card").first();
  await expect(card).toBeVisible({ timeout: 60_000 });
  await expect(card).toContainText("LNG fell 2% or more");            // the condition, in words
  await expect(card.getByTestId("graph-assumptions")).toBeVisible();  // what was assumed is stated before anything runs
  // the result: evidence against the other sessions, a matched comparison, robustness and a held-back final test
  const findings = page.getByTestId("findings").first();
  await expect(findings).toBeVisible({ timeout: 90_000 });
  await page.getByTestId("open-view").last().click();
  const view = page.getByTestId("graph-view");
  await expect(view).toBeVisible();
  await expect(view.getByTestId("forest").first()).toBeVisible();
  await expect(view).toContainText("matched");
  await expect(view).toContainText(/final test|held back/i);
  // it is recorded: the knowledge base lists the finding with a grade, the registry lists the held-back period
  await page.goto("/findings");
  const f = page.getByTestId("finding").first();
  await expect(f).toContainText("LNG");
  await expect(f).toContainText(/confirmed|supported|suggestive|holds|not supported|insufficient/);
  await page.getByTestId("findings-search").fill("zzzz-nothing");
  await expect(page.getByTestId("findings-list")).toContainText("No finding matches that search");
  await page.goto("/registry");
  await expect(page.getByTestId("registry")).toContainText("Final test periods");
  await expect(page.getByTestId("holdouts")).toContainText("LNG");
});

test("the pattern explorer shows every occurrence of an event and what followed", async ({ page }) => {
  await page.goto("/patterns");
  await page.getByTestId("pattern-symbol").selectOption("LNG");
  await page.getByTestId("event-down_day").click();
  await page.getByTestId("pattern-run").click();
  const res = page.getByTestId("pattern-result");
  await expect(res).toContainText("LNG closes 2% or more below the previous close", { timeout: 30_000 });
  await expect(res).toContainText(/It happened \d+ times in [\d,]+ sessions/);
  await expect(res).toContainText("Price paths after the event");
  await expect(res).toContainText("Forward returns by horizon");
  await expect(res).toContainText("Underlying price behaviour only"); // a price pattern is never presented as a strategy
  await res.getByTestId("pattern-investigate").click();
  await page.waitForURL(/\/lab\//);
  await expect(page.getByTestId("msg-user").first()).toContainText("LNG");
});

test("the strategy IDE: describe, edit one section by instruction, and replay a real session", async ({ page }) => {
  await page.goto("/ide");
  await expect(page.getByTestId("code-editor")).toBeVisible();
  await page.getByTestId("ide-describe").click();
  await page.getByLabel("Strategy description").fill("Buy a 0DTE at-the-money TST call when TST is down more than 0.2% from the open and the moon is full, between 10:00 and 15:00 ET. Take profit at 50%, stop at 30%.");
  await page.getByTestId("ide-draft").click();
  const editor = page.getByTestId("code-editor").locator("textarea");
  await expect(editor).toHaveValue(/from_open_pct <= -0\.2/);       // the condition it understood
  await expect(editor).toHaveValue(/time >= hhmm\(10, 0\)/);          // the window is a rule, not an entry time
  await expect(page.getByTestId("problems")).toContainText("the moon is full"); // the part it could not represent is reported, not dropped
  await expect(page.getByTestId("ide")).toContainText("✓ valid");
  // edit ONLY the stop: the changed lines are marked and the reasons are stated
  await page.getByTestId("outline").getByText("stop loss", { exact: false }).first().click();
  await page.getByTestId("ide-instruction").fill("Replace this fixed stop with an ATR-based stop.");
  await page.getByTestId("ide-apply").click();
  await expect(page.getByTestId("ide")).toContainText("paths changed");
  await expect(editor).toHaveValue(/"stop_atr": 1\.5/);
  await expect(editor).toHaveValue(/from_open_pct <= -0\.2/);       // the entry is untouched
  await expect(page.getByTestId("problems")).toContainText("1.5 ATR is used");
  // an instruction that cannot be expressed changes nothing and says so
  await page.getByTestId("ide-instruction").fill("Make it profitable.");
  await page.getByTestId("ide-apply").click();
  await expect(page.getByTestId("problems")).toContainText("Nothing was changed");
  // an error is marked where it is
  await editor.focus();
  await page.keyboard.press("ControlOrMeta+End");
  await page.keyboard.type("x");
  await expect(page.getByTestId("ide")).toContainText("does not parse");
  await page.keyboard.press("Backspace");
  await expect(page.getByTestId("ide")).toContainText("✓ valid");
  // the debugger replays the entry rule on a stored session, tick by tick
  await page.getByTestId("ide-tab-debug").click();
  await page.getByTestId("debug-run").click();
  const dbg = page.getByTestId("debugger");
  await expect(dbg.getByTestId("debug-tick")).toBeVisible({ timeout: 30_000 });
  await expect(dbg.getByTestId("debug-tick")).toContainText("from_open_pct");
  await expect(dbg).toContainText(/evaluations on 1-minute bars/);
});

test("forward validation freezes a study and refuses to judge it on sessions it has already seen", async ({ page, request }) => {
  expect(notebook).not.toBe("");
  await page.goto("/forward");
  await expect(page.getByTestId("forward")).toContainText("No frozen experiment yet");
  const runs = await (await request.get("/api/backtests?kind=graph&limit=20")).json();
  const run = runs.find((r: any) => r.status === "done");
  expect(run).toBeTruthy();
  await page.getByTestId("forward").locator("select").selectOption(run.id);
  await page.getByTestId("forward-freeze").click();
  const d = page.getByTestId("forward-detail");
  await expect(d).toBeVisible();
  await expect(d).toContainText("first session that counts");
  await d.getByTestId("forward-evaluate").click();
  // no session has closed since the configuration was frozen, so there is no forward evidence yet, and it says so
  await expect(d).toContainText("waiting", { timeout: 30_000 });
  await expect(d).toContainText(/No session since activation has closed yet\. The first session that can count is \d{4}-\d{2}-\d{2}/);
  // the frozen configuration cannot be edited through the API either: a change is a new version
  const ex = (await (await request.get("/api/forward")).json()).experiments[0];
  const same = await request.post(`/api/forward/${ex.id}/version`, { data: { config: ex.config } });
  expect(same.status()).toBe(422);
  expect((await same.json()).detail).toContain("identical to the frozen one");
});

test("the data marketplace keeps advertised and verified apart, and prices one strategy's data", async ({ page }) => {
  await page.goto("/data?tab=sources");
  const mk = page.getByTestId("marketplace");
  await expect(mk).toContainText("Where historical options data can come from");
  await expect(mk.getByTestId("source-alpaca").getByTestId("verification")).toContainText("unverified");
  const guide = mk.getByTestId("alpaca-guide");
  await expect(guide).toContainText("Guided connection");
  await expect(guide.locator("li")).toHaveCount(5);
  await expect(mk.getByTestId("source-databento")).toContainText("Advertised");
  await expect(mk.getByTestId("source-databento")).toContainText("Verified on this installation");
  await expect(mk.getByTestId("source-ibkr").getByTestId("verification")).toContainText("no integration");
  // the collector says what can and cannot be recovered after a missed session
  await page.getByRole("tab", { name: /Collector/ }).click();
  await expect(page.getByTestId("collector")).toContainText("Collection only happens while this is running");
  await expect(page.getByTestId("recoverability")).toContainText("After a missed session");
});
