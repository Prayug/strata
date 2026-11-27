/* The V8 usability walkthrough: the brief's thirteen workflows, timed, on the running application and its real stored data
   (dev server on 5199, API on 8765). Not part of the test suite: `npx playwright test -c walkthrough/playwright.config.ts`. */
import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: ".",
  timeout: 15 * 60_000,
  expect: { timeout: 120_000 },
  workers: 1,
  reporter: "list",
  use: { baseURL: process.env.WALK_URL || "http://localhost:5199", viewport: { width: 1440, height: 860 }, channel: process.env.PW_CHANNEL || undefined },
});
