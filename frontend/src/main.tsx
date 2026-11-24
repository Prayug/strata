import "./index.css";
import { QueryClientProvider } from "@tanstack/react-query";
import { StrictMode, Suspense, lazy } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { Toaster } from "sonner";
import { Loading } from "@/components/common";
import { Shell } from "@/components/Shell";
import { TooltipProvider } from "@/components/ui";
import { queryClient } from "@/lib/api";

const Overview = lazy(() => import("@/pages/Overview"));
const Builder = lazy(() => import("@/pages/builder/Builder"));
const Results = lazy(() => import("@/pages/results/Results"));
const Replay = lazy(() => import("@/pages/Replay"));
const Optimize = lazy(() => import("@/pages/Optimize"));
const Research = lazy(() => import("@/pages/Research"));
const Portfolio = lazy(() => import("@/pages/Portfolio"));
const Strategies = lazy(() => import("@/pages/Strategies"));
const DataPage = lazy(() => import("@/pages/DataPage"));

const S = (el: React.ReactNode) => <Suspense fallback={<Loading />}>{el}</Suspense>;

const router = createBrowserRouter([
  {
    path: "/",
    element: <Shell />,
    children: [
      { index: true, element: S(<Overview />) },
      { path: "builder", element: S(<Builder />) },
      { path: "results", element: S(<Results />) },
      { path: "results/:id", element: S(<Results />) },
      { path: "replay", element: S(<Replay />) },
      { path: "replay/:id/:tid", element: S(<Replay />) },
      { path: "optimize", element: S(<Optimize />) },
      { path: "optimize/:id", element: S(<Optimize />) },
      { path: "research", element: S(<Research />) },
      { path: "portfolio", element: S(<Portfolio />) },
      { path: "portfolio/:id", element: S(<Portfolio />) },
      { path: "strategies", element: S(<Strategies />) },
      { path: "data", element: S(<DataPage />) },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <TooltipProvider delayDuration={250}>
        <RouterProvider router={router} />
        <Toaster theme="dark" position="bottom-right" toastOptions={{ style: { background: "#1a202b", border: "1px solid #2b3442", color: "#d6dce5", fontSize: 12, borderRadius: 3 } }} />
      </TooltipProvider>
    </QueryClientProvider>
  </StrictMode>,
);
