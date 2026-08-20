import { Activity, Github, LayoutDashboard } from "lucide-react";
import Link from "next/link";
import type { ReactNode } from "react";

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="sticky top-0 z-20 border-b border-white/8 bg-slate-950/85 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-5">
          <Link href="/" className="flex items-center gap-2 font-bold tracking-tight"><span className="grid size-9 place-items-center rounded-xl bg-cyan-300 text-slate-950"><Activity size={19} /></span>BugRisk AI</Link>
          <nav className="flex items-center gap-1 text-sm" aria-label="Primary navigation">
            <Link className="rounded-lg px-3 py-2 text-slate-300 hover:bg-white/5 hover:text-white" href="/dashboard"><LayoutDashboard className="mr-1.5 inline" size={15} />Dashboard</Link>
            <Link className="rounded-lg px-3 py-2 text-slate-300 hover:bg-white/5 hover:text-white" href="/models"><Activity className="mr-1.5 inline" size={15} />Model</Link>
            <a className="hidden rounded-lg px-3 py-2 text-slate-300 hover:bg-white/5 hover:text-white sm:block" href="https://github.com" target="_blank" rel="noreferrer"><Github className="mr-1.5 inline" size={15} />GitHub</a>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-5 py-8">{children}</main>
    </div>
  );
}

