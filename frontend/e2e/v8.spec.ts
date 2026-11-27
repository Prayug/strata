/* End-to-end for V8, through the browser, on the seeded fixture store (backend/tests/e2e_seed.py): TST fell 3% or more on
   two of its eight sessions; the session after the first has option trade prints for the 96 call. The answers are known. */
import { expect, test, type Page } from "@playwright/test";

const ask = async (page: Page, text: string) => {
  const box = page.getByTestId("composer");
  await box.click();
  await box.fill(text);
  await box.press("Enter");
  await expect(page.getByTestId("question").last()).toHaveText(text);
};

test.describe.configure({ mode: "serial" });
let notebook = "";

test("a question is answered first, in words, and every count opens the days behind it", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto("/");
  await expect(page.getByTestId("home")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Ask what the market has done before." })).toBeVisible();
  const box = page.getByTestId("home-ask");
  await box.fill("How often does TST rise 1% from the open after falling 3% or more the previous trading day?");
  await box.press("Enter");
  await expect(page).toHaveURL(/\/ask\/[0-9a-f]+/);
  notebook = page.url();
  const head = page.getByTestId("headline").last();
  await expect(head).toContainText("on 1 of 2 days when the previous session fell 3% or more (50.0%)", { timeout: 60_000 });
  await expect(page.getByTestId("viz-frequency")).toBeVisible();
  // the count is a door: the one day it happened, with a replay
  await page.getByTestId("figure").filter({ hasText: "Days it happened" }).click();
  const sheet = page.getByTestId("explore-sheet");
  await expect(sheet).toContainText("Days it happened");
  await expect(sheet.getByTestId("obs-table").locator("tbody tr")).toHaveCount(1);
  await page.keyboard.press("Escape");
  await expect(sheet).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("a follow-up changes one part, says so, can be undone, and a comparison is explained in words", async ({ page }) => {
  await page.goto(notebook);
  await ask(page, "Change the previous-day decline to 3.8% and compare the results.");
  const change = page.getByTestId("change").last();
  await expect(change).toContainText("Now: the previous session fell 3.8% or more (was 3%). Everything else stays the same.");
  await expect(page.getByTestId("headline").last()).toContainText("When the previous session fell 3.8% or more, it happened more often", { timeout: 60_000 });
  await expect(page.getByTestId("undo")).toBeVisible();
  // earlier answers fold to their finding; the newest is open
  await expect(page.getByTestId("folded-result").first()).toBeVisible();
});

test("a strategy is a sentence with values that can be changed, and the change is said before it runs", async ({ page }) => {
  await page.goto(notebook);
  await ask(page, "Buy ATM TST 0DTE calls at the open after a 3% down day using option trade prices, exit at 15:55");
  await expect(page.getByTestId("headline").last()).toContainText("Over 1 trade", { timeout: 60_000 });
  const sentence = page.getByTestId("strategy-sentence");
  await expect(sentence).toContainText("Buy TST calls with 0 days to expiration");
  await sentence.getByTestId("slot-exit").click();
  await page.locator('input[type="time"]').fill("14:55");
  await page.getByRole("button", { name: "Set", exact: true }).click();
  await expect(page.getByTestId("pending-changes")).toContainText("Changed the exit time from 3:55 PM Eastern to 2:55 PM Eastern.");
  await page.getByTestId("run-changes").click();
  await expect(page.getByTestId("version-change")).toContainText("Changed the exit time", { timeout: 60_000 });
  // one real trade, explained from its own prices and the actual contract multiplier
  await ask(page, "Explain this like I'm new to options");
  await expect(page.getByTestId("content-answer").last()).toContainText("One contract covers 100 shares", { timeout: 30_000 });
  await expect(page.getByTestId("viz-payoff")).toContainText("Illustration, not historical data");
});

test("the detail level is remembered, and the technical level keeps the full report", async ({ page }) => {
  await page.goto(notebook);
  await page.getByTestId("level-technical").last().click();
  await expect(page.getByTestId("level-technical").last()).toHaveAttribute("aria-pressed", "true");
  await page.reload();
  await expect(page.getByTestId("level-technical").last()).toHaveAttribute("aria-pressed", "true");
  await page.getByTestId("level-answer").last().click();
});

test("on a phone the answer is one readable column", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto(notebook);
  await expect(page.getByTestId("headline").last()).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(1);
});
