import { Activity, LayoutDashboard, ScanSearch } from "lucide-react";
import Link from "next/link";
import type { ReactNode } from "react";

import { LogoutButton } from "@/components/logout-button";

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen text-ink">
      <header className="sticky top-0 z-20 border-b-2 border-ink bg-canvas/90 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-5">
          <Link href="/" className="flex items-center gap-2.5 font-mono text-sm font-black tracking-tight"><span className="grid size-9 place-items-center rounded-xl border-2 border-ink bg-pink shadow-[3px_3px_0_#191919]"><ScanSearch size={18} /></span>BUGRISK<span className="text-[#a13363]">.AI</span></Link>
          <nav className="flex items-center gap-1 font-mono text-[11px] font-bold uppercase" aria-label="Primary navigation">
            <Link aria-label="Dashboard" className="rounded-full px-3 py-2 hover:bg-butter" href="/dashboard"><LayoutDashboard className="mr-1.5 inline" size={14} /><span className="hidden sm:inline">Dashboard</span></Link>
            <Link aria-label="Model card" className="rounded-full px-3 py-2 hover:bg-lilac" href="/models"><Activity className="mr-1.5 inline" size={14} /><span className="hidden sm:inline">Model card</span></Link>
            <LogoutButton />
          </nav>
        </div>
      </header>
      <main className="page-enter mx-auto max-w-7xl px-5 py-8 sm:py-10">{children}</main>
    </div>
  );
}
