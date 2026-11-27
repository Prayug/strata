/* End-to-end: a person asks a question, follows up, and checks the evidence, all through the browser.
   The fixture store has known answers (backend/tests/e2e_seed.py): TST fell 3% or more on two days;
   the next session reached +1% from its open once. Only one of those two sessions has option trade prints. */
import { expect, test, type Page } from "@playwright/test";

const ask = async (page: Page, text: string) => {
  const box = page.getByTestId("composer");
  await box.click();
  await box.fill(text);
  await box.press("Enter");
  await expect(page.getByTestId("msg-user").last()).toHaveText(text);
};
const idle = async (page: Page) => {
  // every run has finished and been turned into a result
  await expect(page.getByText("Reading the request…")).toHaveCount(0);
  await expect(page.locator('[data-testid="pipeline"] .animate-spin')).toHaveCount(0, { timeout: 60_000 });
};

test("a market question is answered with a report whose numbers open their observations", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto("/lab");
  await expect(page.getByRole("heading", { name: "What do you want to find out about the market?" })).toBeVisible();
  await ask(page, "How often does TST rise 1% after a 3% down day?");
  await expect(page).toHaveURL(/\/lab\/[0-9a-f]+/);
  const result = page.getByTestId("result-single");
  await expect(result).toBeVisible();
  await expect(result).toContainText("1 of the 2 sessions");
  // how it was read is shown, with the assumption labelled as an assumption
  await expect(page.getByTestId("conversation")).toContainText("previous session fell 3% or more");
  await expect(page.getByTestId("conversation")).toContainText("assumed");
  // the report: headline, then the sessions behind it
  await expect(page.getByRole("heading", { name: "How often does TST rise 1% after a 3% down day?" })).toBeVisible();
  const headline = page.getByTitle("Show the observations behind this number").first();
  await expect(headline).toContainText("How often");
  await expect(headline).toContainText("50.0%");
  await headline.click();
  await expect(page.getByTestId("tab-observations")).toHaveAttribute("aria-selected", "true");
  await expect(page.locator("table.t tbody tr")).toHaveCount(1); // the one session that reached the target
  await page.getByRole("button", { name: "Qualifying", exact: true }).click();
  await expect(page.locator("table.t tbody tr")).toHaveCount(2);
  // keyboard: 2 opens the calendar, the qualifying days are marked, a click replays the day
  await page.keyboard.press("2");
  await expect(page.getByTestId("calendar")).toBeVisible();
  await expect(page.locator('button.day[data-status="target"]')).toHaveCount(1);
  await expect(page.locator('button.day[data-status="neither"]')).toHaveCount(1);
  await page.locator('button.day[data-status="target"]').click();
  await expect(page.getByTestId("calendar")).toContainText("Target reached");
  await expect(page.getByTestId("calendar")).toContainText("How each rule evaluated");
  // methodology and robustness are part of the same report
  await page.keyboard.press("5");
  await expect(page.getByTestId("tab-robustness")).toHaveAttribute("aria-selected", "true");
  await page.keyboard.press("6");
  await expect(page.getByTestId("tab-methodology")).toHaveAttribute("aria-selected", "true");
  await expect(page.locator("body")).toContainText("Data sources and coverage");
  await expect(page.locator("body")).toContainText("Assumptions made");
  expect(errors).toEqual([]);
});

test("follow-ups reuse the question: sessions, option trade prices, skipped signals, export", async ({ page, request }) => {
  await page.goto("/lab");
  await ask(page, "How often does TST rise 1% after a 3% down day?");
  await expect(page.getByTestId("result-single")).toBeVisible();
  await ask(page, "Show every qualifying historical session.");
  await expect(page.getByTestId("tab-observations")).toHaveAttribute("aria-selected", "true");
  await expect(page.locator("table.t tbody tr")).toHaveCount(2);

  await ask(page, "Test ATM 0DTE calls on those days using historical option trade prices.");
  await idle(page);
  const opt = page.getByTestId("result-single").last();
  await expect(opt).toContainText("only 1 of them has usable observed option prices");
  await expect(opt).toContainText("One trade is an observation, not evidence");
  await expect(opt).toContainText("Options · trade prices");
  // the report says what the prices are, and never shows a bid/ask that was not observed
  await expect(page.locator("body")).toContainText("Priced on option trade prints");
  await page.keyboard.press("Escape"); // leave the composer, so the digit is a shortcut and not text
  await page.keyboard.press("3");
  await expect(page.getByTestId("tab-observations")).toHaveAttribute("aria-selected", "true");
  await expect(page.locator("table.t tbody tr")).toHaveCount(1);
  await expect(page.locator("table.t tbody tr").first()).toContainText("trade print");

  await ask(page, "Show me every skipped signal and why it was skipped.");
  await expect(page.getByTestId("result-skipped")).toContainText("1 qualifying session");
  await expect(page.locator("table.t tbody tr")).toHaveCount(1);
  await expect(page.locator("table.t tbody tr").first()).toContainText("no option trade data is stored for the session");

  await ask(page, "Export the complete research report.");
  const exp = page.getByTestId("result-export");
  await expect(exp).toBeVisible();
  const href = await exp.locator('a[href*="format=html"]').getAttribute("href");
  const html = await (await request.get(href!)).text();
  expect(html).toContain("Test ATM 0DTE calls");
  expect(html).toContain("trade print");
  expect(html.toLowerCase()).not.toContain("api_key");
  const csv = await (await request.get((await exp.locator('a[href*="what=skipped"]').getAttribute("href"))!)).text();
  expect(csv.split("\n").filter(Boolean).length).toBe(2); // header + the one skipped signal
});

test("review mode asks the one ambiguous detail and waits for confirmation", async ({ page }) => {
  await page.goto("/lab");
  await page.getByRole("button", { name: "Review first" }).click();
  await ask(page, "Buy TST 0DTE calls at the open with a 1% target after a 3% down day");
  const q = page.getByTestId("question");
  await expect(q).toContainText("You said target 1%. Do you mean a 1% move in TST or a 1% gain on the option?");
  await expect(page.getByTestId("result-single")).toHaveCount(0); // nothing has run
  await q.getByRole("button", { name: "1% move in TST" }).click();
  // the interpretation is editable in place, and can be checked on a real stored session first
  await expect(page.getByTestId("setup")).toBeVisible();
  await expect(page.getByTestId("setup")).toContainText("An option strategy");
  await page.getByTestId("preview").getByRole("button", { name: /Show me a day where it fires/ }).click();
  await expect(page.getByTestId("preview")).toContainText("How each rule evaluated");
  await page.getByTestId("confirm").getByRole("button", { name: /run it/i }).click();
  await idle(page);
  await expect(page.getByTestId("result-single").or(page.getByTestId("decision")).first()).toBeVisible();
});

test("a part that cannot be represented is named, and nothing is put in its place", async ({ page }) => {
  await page.goto("/lab");
  await ask(page, "Buy TST 0DTE calls at 10:00 ET after a 3% down day when dealers are short gamma and the realized skew is inverted");
  const conv = page.getByTestId("conversation");
  await expect(conv).toContainText("Part of the request could not be represented");
  await expect(conv).toContainText("when dealers are short gamma and the realized skew is inverted");
  await expect(conv).toContainText("Nothing was substituted for it");
  await expect(page.getByTestId("result-single")).toHaveCount(0);
  await expect(page.getByTestId("confirm").getByRole("button", { name: "Run without that part" })).toBeVisible();
});

test("pinned results are compared with their differences disclosed", async ({ page }) => {
  await page.goto("/lab");
  await ask(page, "How often does TST rise 1% after a 3% down day?");
  await expect(page.getByTestId("result-single")).toBeVisible();
  await page.getByTestId("result-single").getByRole("button", { name: /Pin to compare/ }).click();
  await ask(page, "How often does TST rise 0.5% after a 3% down day?");
  await expect(page.getByTestId("result-single")).toHaveCount(2);
  await page.getByTestId("result-single").last().getByRole("button", { name: /Pin to compare/ }).click();
  await page.keyboard.press("Escape");
  await page.getByTestId("open-compare").click();
  const cmp = page.getByTestId("compare");
  await expect(cmp).toBeVisible();
  await expect(cmp.locator("table.t").first()).toContainText("underlying prices");
  await expect(cmp).toContainText("On the same dates");
});

test("settings: credentials are write-only and the Alpaca connection is guided", async ({ page }) => {
  await page.goto("/settings#alpaca");
  await expect(page.getByText("Connect a free Alpaca account")).toBeVisible();
  const inputs = page.locator("#alpaca input");
  await expect(inputs).toHaveCount(2);
  for (const i of [0, 1]) await expect(inputs.nth(i)).toHaveAttribute("type", "password");
  await expect(page.locator("#alpaca").getByRole("button", { name: /Test connection|Save and test/ })).toBeDisabled();
  // the settings payload never carries a secret value, only whether one is set
  const body = await (await page.request.get("/api/settings")).json();
  for (const s of Object.values(body.secrets) as any[]) expect(Object.keys(s).sort()).toEqual(["hint", "label", "set", "source"]);
  expect(body.secret_store.backend).toBe("file");
  // the data page lists each dataset with its provider and observation type
  await page.goto("/data");
  await page.getByRole("button", { name: /advanced tools/ }).click(); // provenance is an advanced tool: one click shows it
  await page.getByRole("tab", { name: /Datasets/ }).click();
  await expect(page.locator("table.tbl tbody tr").filter({ hasText: "option trade prints" })).toHaveCount(1);
});
