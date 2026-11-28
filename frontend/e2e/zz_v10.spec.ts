/* End-to-end for V10, on the seeded fixture store (backend/tests/e2e_seed.py, ticker TST). The file is named to run last: it
   switches on the go-ahead before a request with several parts, which the specs written before V10 run without.

   A simple question is answered at once and says how far it was verified; a request with several parts shows how it was read
   and waits; a part of the reading can be changed with a click; a go-ahead typed in a person's own words runs it; what the
   stored data cannot answer is declined with the reason. */
import { expect, test, type Page } from "@playwright/test";

const ask = async (page: Page, text: string) => {
  const box = page.getByTestId("composer");
  await box.click();
  await box.fill(text);
  await box.press("Enter");
  await expect(page.getByTestId("question").last()).toHaveText(text);
};

const start = async (page: Page, text: string) => {
  await page.goto("/");
  const box = page.getByTestId("home-ask");
  await box.fill(text);
  await box.press("Enter");
  await expect(page).toHaveURL(/\/ask\/[0-9a-f]+/);
};

const COMPLEX = "After a 3% down day, how often does TST close above its open on Mondays?";
const READING = "On TST, when the previous session fell 3% or more and on Mondays: how often it closes above its open";

test.describe.configure({ mode: "serial" });

test.beforeAll(async ({ request }) => {
  const r = await request.put("/api/settings", { data: { settings: { "ai.confirm_complex": true } } });
  expect(r.ok()).toBeTruthy();
});

test.afterAll(async ({ request }) => {
  await request.put("/api/settings", { data: { settings: { "ai.confirm_complex": false } } });
});

test("a simple question is answered at once, and says it was fully verified", async ({ page }) => {
  await start(page, "How often does TST close above its open?");
  await expect(page.getByTestId("headline").last()).toHaveText("TST closes above its open on 2 of 7 days (28.6%).", { timeout: 60_000 });
  const check = page.getByTestId("check").last();
  await expect(check).toHaveAttribute("data-status", "same");
  await expect(check).toContainText("Fully verified");
  await expect(page.getByTestId("reading").last()).toContainText("On TST: how often it closes above its open");
  await expect(page.getByTestId("confirm-reading")).toHaveCount(0);
});

test("a request with several parts shows how it was read and waits; a click on a part changes it", async ({ page }) => {
  await start(page, COMPLEX);
  const block = page.getByTestId("confirm-reading").last();
  await expect(block).toContainText("Check this reading before it runs", { timeout: 60_000 });
  await expect(block.getByTestId("reading-line")).toContainText(READING);
  await expect(block.getByTestId("run-as-read")).toHaveText("Yes, run it");
  await expect(block).toContainText("Nothing has run.");
  await expect(page.getByTestId("headline")).toHaveCount(0);
  // the weekday is one of the parts: clicking it offers to remove it, and choosing that sends the change
  await block.getByTestId("reading-condition").filter({ hasText: "on Mondays" }).click();
  const menu = block.getByTestId("reading-menu");
  await expect(menu).toContainText("Change this condition");
  await menu.getByTestId("reading-choice").filter({ hasText: "Remove this condition" }).click();
  await expect(page.getByTestId("question").last()).toHaveText("remove the weekday condition");
  // the changed reading is shown and waits again (since the held-out run: a correction never runs a reading by itself)
  const again = page.getByTestId("confirm-reading").last();
  await expect(again.getByTestId("reading-line")).toContainText("On TST, when the previous session fell 3% or more: how often it closes above its open");
  await expect(again.getByTestId("reading-line")).not.toContainText("Mondays");
  await expect(page.getByTestId("headline")).toHaveCount(0);
  await again.getByTestId("run-as-read").click();
  await expect(page.getByTestId("headline").last()).toHaveText("TST closes above its open on 2 of 2 days when the previous session fell 3% or more (100.0%).", { timeout: 60_000 });
  await expect(page.getByTestId("check").last()).toContainText("Fully verified");
});

test("a go-ahead typed in a person's own words runs the reading that was shown", async ({ page }) => {
  await start(page, COMPLEX);
  await expect(page.getByTestId("confirm-reading").last()).toBeVisible({ timeout: 60_000 });
  await ask(page, "yep that's what I meant");
  await expect(page.getByTestId("headline").last()).toContainText("No day in the stored TST history matched your rules", { timeout: 60_000 });
  await expect(page.getByTestId("reading").last()).toContainText(READING);
  // with nothing waiting, a bare yes says so instead of being taken for a change
  await ask(page, "yes");
  await expect(page.getByTestId("note").last()).toContainText("Nothing is waiting to be run");
});

test("what the stored data cannot answer is declined with the reason", async ({ page }) => {
  await start(page, "what does TST do on fed days");
  const no = page.getByTestId("declined").last();
  await expect(no).toContainText("needs a calendar of scheduled events", { timeout: 60_000 });
  await expect(no).toContainText("Nothing was run");
  await expect(page.getByTestId("headline")).toHaveCount(0);
});
