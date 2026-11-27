/* End-to-end for V9, on the seeded fixture store (backend/tests/e2e_seed.py, ticker TST): an answer says whether its numbers
   were recalculated independently; a question about a ticker with no history is declined with the closest question that
   can be answered, which runs when clicked; "what would you investigate next" comes from the result; a strategy is sized in
   a trader's terms in its sentence, and the change is said before it runs. */
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

test("an answer says whether its numbers were recalculated independently", async ({ page }) => {
  await page.goto("/");
  const box = page.getByTestId("home-ask");
  await box.fill("How often does TST close above its open?");
  await box.press("Enter");
  await expect(page).toHaveURL(/\/ask\/[0-9a-f]+/);
  notebook = page.url();
  await expect(page.getByTestId("headline").last()).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId("check").last()).toContainText("Recalculated independently from the stored bars");
});

test("what to investigate next comes from the result, and each suggestion asks itself", async ({ page }) => {
  await page.goto(notebook);
  await ask(page, "what would you investigate next?");
  const steps = page.getByTestId("next-step");
  await expect(steps.first()).toBeVisible({ timeout: 30_000 });
  expect(await steps.count()).toBeGreaterThan(0);
});

test("a ticker with no history is declined, with the closest question offered and run on a click", async ({ page }) => {
  await page.goto(notebook);
  await ask(page, "how often does NVDA close above its open?");
  await expect(page.getByTestId("declined").last()).toContainText("No price history for NVDA");
  const alt = page.getByTestId("closest-question").last();
  await expect(alt).toContainText("TST");
  await alt.click();
  await expect(page.getByTestId("question").last()).toContainText("TST");
});

test("a strategy is sized in a trader's terms in its sentence, and the change is said before it runs", async ({ page }) => {
  await page.goto(notebook);
  await ask(page, "Buy ATM TST 0DTE calls at the open after a 3% down day using option trade prices, exit at 15:55");
  const sentence = page.getByTestId("strategy-sentence").last();
  await expect(sentence).toContainText("Size each trade at 1 contract", { timeout: 60_000 });
  await sentence.getByTestId("slot-size").click();
  const pop = page.getByTestId("size-edit");
  await pop.locator("select").selectOption("dollars");
  await pop.locator('input[type="number"]').first().fill("500");
  await pop.getByRole("button", { name: "Set", exact: true }).click();
  await expect(page.getByTestId("pending-changes")).toContainText("Changed the size of each trade from 1 contract to $500 of premium");
});
