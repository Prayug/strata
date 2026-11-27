/* The thirteen workflows of the V8 brief, done the way a person would: from the landing page, without opening settings,
   the classic workspace or documentation. Each step records its clicks, what had to be known, and the seconds until its
   answer was on screen; screenshots go to docs/v8_evidence/screens. Written for the stored SPY/QQQ/IWM history. */
import { expect, test, type Page } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const OUT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../docs/v8_evidence");
const SHOTS = path.join(OUT, "screens");
type Row = { n: number; workflow: string; how: string; clicks: number; typed: string | null; knowledge: string; seconds: number; result: string };
const rows: Row[] = [];

async function shot(page: Page, name: string) {
  await page.screenshot({ path: path.join(SHOTS, `${name}.png`), fullPage: false });
}
async function type(page: Page, text: string) {
  const box = page.getByTestId("composer");
  await box.click();
  await box.fill(text);
  await box.press("Enter");
}
async function timed<T>(fn: () => Promise<T>): Promise<number> {
  const t0 = Date.now();
  await fn();
  return Math.round((Date.now() - t0) / 100) / 10;
}
const lastHeadline = (page: Page) => page.getByTestId("headline").last();

test("the thirteen workflows, timed, at laptop size", async ({ page }) => {
  fs.mkdirSync(SHOTS, { recursive: true });
  await page.goto("/");
  await expect(page.getByTestId("home")).toBeVisible();
  await shot(page, "laptop-1440x860-01-landing");

  // 1. ask whether the opening candle predicts the rest of SPY's day
  let s = await timed(async () => {
    await page.getByTestId("starter").filter({ hasText: "first five-minute candle" }).click();
    await expect(lastHeadline(page)).toContainText("matched the direction of the rest of the session", { timeout: 180_000 });
  });
  await shot(page, "laptop-1440x860-02-first-candle-answer");
  rows.push({ n: 1, workflow: "Ask whether the opening candle predicts the rest of SPY's day", how: "clicked the starter question", clicks: 1, typed: null, knowledge: "none", seconds: s,
    result: (await lastHeadline(page).textContent())?.slice(0, 220) || "" });

  // 2. change the candle length
  s = await timed(async () => { await type(page, "use 15 min"); await expect(page.getByTestId("change").last()).toContainText("15-minute"); await expect(lastHeadline(page)).toContainText("15-minute"); });
  rows.push({ n: 2, workflow: "Change the candle length", how: "typed a follow-up", clicks: 1, typed: "use 15 min", knowledge: "none", seconds: s, result: (await page.getByTestId("change").last().textContent()) || "" });

  // 3. correct the ticker informally
  s = await timed(async () => { await type(page, "nah do qqq"); await expect(page.getByTestId("change").last()).toContainText("QQQ"); await expect(lastHeadline(page)).toContainText("15-minute"); });
  rows.push({ n: 3, workflow: "Correct the ticker informally", how: "typed a follow-up", clicks: 1, typed: "nah do qqq", knowledge: "the ticker", seconds: s, result: (await page.getByTestId("change").last().textContent()) || "" });

  // 4. add a previous-day condition
  s = await timed(async () => { await type(page, "only when yesterday was red"); await expect(page.getByTestId("change").last()).toContainText("previous session closed down"); await expect(lastHeadline(page)).toBeVisible(); });
  await shot(page, "laptop-1440x860-03-condition-added");
  rows.push({ n: 4, workflow: "Add a previous-day condition", how: "typed a follow-up", clicks: 1, typed: "only when yesterday was red", knowledge: "none", seconds: s, result: (await page.getByTestId("change").last().textContent()) || "" });

  // 5. inspect the historical observations behind the answer
  s = await timed(async () => {
    await page.getByTestId("figure").filter({ hasText: "Days the candle was wrong" }).last().click();
    await expect(page.getByTestId("obs-table")).toBeVisible();
  });
  await shot(page, "laptop-1440x860-04-observations");
  const obs = (await page.getByTestId("explore-sheet").locator("text=/of [0-9,]+/").first().textContent()) || "";
  await page.keyboard.press("Escape");
  rows.push({ n: 5, workflow: "Inspect the historical observations", how: "clicked a number in the answer", clicks: 1, typed: null, knowledge: "none", seconds: s, result: `list opened: ${obs}` });

  // 6. convert the idea into an options strategy
  s = await timed(async () => { await type(page, "what if I bought calls instead?"); await expect(page.getByTestId("strategy-sentence").last()).toBeVisible(); await expect(lastHeadline(page)).toContainText("trade", { timeout: 300_000 }); });
  await shot(page, "laptop-1440x860-05-strategy");
  rows.push({ n: 6, workflow: "Convert the idea into an options strategy", how: "typed a follow-up", clicks: 1, typed: "what if I bought calls instead?", knowledge: "what a call is", seconds: s,
    result: (await lastHeadline(page).textContent())?.slice(0, 220) || "" });

  // 7. modify its entry and exit conditions (in the sentence)
  s = await timed(async () => {
    const sent = page.getByTestId("strategy-sentence").last();
    await sent.getByTestId("slot-exit").click();
    await page.locator('input[type="time"]').fill("14:30");
    await page.getByRole("button", { name: "Set", exact: true }).click();
    await expect(page.getByTestId("pending-changes")).toContainText("exit time");
  });
  await shot(page, "laptop-1440x860-06-sentence-edit");
  rows.push({ n: 7, workflow: "Modify its entry and exit conditions", how: "clicked the exit time in the sentence and set a new one", clicks: 3, typed: "14:30 (time field)", knowledge: "none", seconds: s,
    result: (await page.getByTestId("pending-changes").textContent()) || "" });

  // 8. run the historical backtest
  const results = page.locator('[data-testid="answer"], [data-testid="folded-result"]');  // every result is one of these: open or folded
  const before = await results.count();
  s = await timed(async () => {
    await page.getByTestId("run-changes").click();
    await expect(page.getByTestId("version-change")).toBeVisible({ timeout: 300_000 });
    await expect(results).toHaveCount(before + 1, { timeout: 300_000 }); // the new version's own result, not the previous one
  });
  rows.push({ n: 8, workflow: "Run the historical backtest", how: "clicked Run with these changes", clicks: 1, typed: null, knowledge: "none", seconds: s, result: (await lastHeadline(page).textContent())?.slice(0, 220) || "" });

  // 9. understand why its largest loss occurred
  s = await timed(async () => { await type(page, "why did the biggest loss happen?"); await expect(lastHeadline(page)).toContainText("biggest loss"); });
  await shot(page, "laptop-1440x860-07-biggest-loss");
  rows.push({ n: 9, workflow: "Understand why its largest loss occurred", how: "typed a question", clicks: 1, typed: "why did the biggest loss happen?", knowledge: "none", seconds: s,
    result: (await page.getByTestId("content-answer").last().textContent())?.slice(0, 300) || "" });

  // 10. compare two strategy variants
  s = await timed(async () => { await type(page, "compare both"); await expect(lastHeadline(page)).toContainText(/made (more money|the same)|Both made/, { timeout: 300_000 }); });
  await shot(page, "laptop-1440x860-08-compare");
  rows.push({ n: 10, workflow: "Compare two strategy variants", how: "typed a follow-up", clicks: 1, typed: "compare both", knowledge: "none", seconds: s, result: (await lastHeadline(page).textContent())?.slice(0, 260) || "" });

  // 11. ask whether the difference might just be luck
  s = await timed(async () => { await type(page, "is the difference just luck?"); await expect(lastHeadline(page)).toContainText(/luck|chance|variation/); });
  rows.push({ n: 11, workflow: "Ask whether the difference might just be luck", how: "typed a question", clicks: 1, typed: "is the difference just luck?", knowledge: "none", seconds: s,
    result: (await page.getByTestId("content-answer").last().textContent())?.slice(0, 300) || "" });

  // 12. inspect the actual option prices used
  s = await timed(async () => {
    await type(page, "show me the option prices used");
    await expect(lastHeadline(page)).toContainText("Every price used");
    await page.getByTestId("figure").filter({ hasText: "option prices" }).last().click();
    await expect(page.getByTestId("obs-table")).toBeVisible();
  });
  await shot(page, "laptop-1440x860-09-option-prices");
  await page.keyboard.press("Escape");
  rows.push({ n: 12, workflow: "Inspect the actual option prices used", how: "typed a request, clicked the count", clicks: 2, typed: "show me the option prices used", knowledge: "premium (the option's price)", seconds: s,
    result: "trade list with premium paid and received per share, contract, exit reason" });

  // 13. return to the original investigation
  s = await timed(async () => {
    await page.getByRole("navigation", { name: "Questions in this notebook" }).getByRole("link").first().click();
    const first = page.getByTestId("turn").first();
    await first.getByRole("button", { name: "Unfold" }).click();
    await first.getByTestId("folded-result").first().click();
    await expect(first.getByTestId("headline")).toContainText("first 5-minute candle");
  });
  await shot(page, "laptop-1440x860-10-back-to-the-start");
  rows.push({ n: 13, workflow: "Return to the original investigation", how: "clicked the first question in the notebook index, unfolded it, opened its answer", clicks: 3, typed: null, knowledge: "none", seconds: s,
    result: (await page.getByTestId("turn").first().getByTestId("headline").textContent())?.slice(0, 200) || "" });

  fs.writeFileSync(path.join(OUT, "usability_walkthrough.json"), JSON.stringify({ when: new Date().toISOString(), viewport: "1440x860", notebook: page.url(), rows }, null, 1));
});

test("screens at desktop and phone sizes", async ({ browser }) => {
  for (const [name, w, h] of [["desktop-1920x1080", 1920, 1080], ["mobile-375x812", 375, 812]] as const) {
    const ctx = await browser.newContext({ viewport: { width: w, height: h }, ...(w < 500 ? { isMobile: true, hasTouch: true, deviceScaleFactor: 2 } : {}) });
    const page = await ctx.newPage();
    await page.goto("/");
    await expect(page.getByTestId("home")).toBeVisible();
    await page.screenshot({ path: path.join(SHOTS, `${name}-01-landing.png`) });
    const nb = JSON.parse(fs.readFileSync(path.join(OUT, "usability_walkthrough.json"), "utf8")).notebook;
    await page.goto(nb);
    await expect(page.getByTestId("headline").last()).toBeVisible();
    await page.getByTestId("turn").first().scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(SHOTS, `${name}-02-notebook.png`) });
    if (w < 500) {
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
      expect(overflow).toBeLessThanOrEqual(1);
    }
    await ctx.close();
  }
});
