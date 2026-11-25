/* Browser end-to-end tests. They build nothing themselves: `pnpm build` first, then `pnpm e2e`.
   One server is started on a throw-away data directory seeded with a small fixture store
   (see backend/tests/e2e_seed.py), with downloads and the OS keychain switched off. */
import { defineConfig } from "@playwright/test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const PORT = Number(process.env.E2E_PORT || 8799);
const backend = path.resolve(__dirname, "../backend");
const venv = path.join(backend, ".venv/bin/python");
const python = process.env.STRATA_PYTHON || (fs.existsSync(venv) ? venv : "python");
const dataDir = process.env.E2E_DATA_DIR || fs.mkdtempSync(path.join(os.tmpdir(), "strata-e2e-"));
process.env.E2E_DATA_DIR = dataDir; // test workers inherit it, so a test can point the app at a fixture file in that directory

export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    viewport: { width: 1480, height: 920 },
    // PW_CHANNEL=chrome uses the Chrome that is already installed instead of a downloaded browser
    channel: process.env.PW_CHANNEL || undefined,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: {
    command: `"${python}" -m tests.e2e_seed "${dataDir}" && "${python}" -m strata.cli serve --port ${PORT}`,
    cwd: backend,
    url: `http://127.0.0.1:${PORT}/api/health`,
    timeout: 120_000,
    reuseExistingServer: false,
    env: { ...process.env, STRATA_DATA_DIR: dataDir, STRATA_OFFLINE: "1", STRATA_SECRET_STORE: "file", STRATA_MAX_WORKERS: "2", STRATA_FRONTEND_DIST: path.resolve(__dirname, "dist") } as Record<string, string>,
  },
});
