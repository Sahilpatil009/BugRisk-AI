"use client";

import { AlertCircle, ArrowRight, Clock3, FolderGit2, GitBranch, Loader2, ShieldCheck } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

import { RiskBadge } from "@/components/risk-badge";
import { Card, Skeleton } from "@/components/ui";
import { api, percent } from "@/lib/api";
import type { Analysis, Repository } from "@/lib/types";

export function DashboardClient() {
  const [repositories, setRepositories] = useState<Repository[]>([]);
  const [analyses, setAnalyses] = useState<Analysis[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api<Repository[]>("/repositories"), api<Analysis[]>("/analyses")])
      .then(([repoData, analysisData]) => { setRepositories(repoData); setAnalyses(analysisData); })
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div aria-label="Loading dashboard" className="grid gap-5 md:grid-cols-3"><Skeleton className="h-44" /><Skeleton className="h-44" /><Skeleton className="h-44" /></div>;
  if (error) return <Card className="flex items-start gap-3 bg-pink p-6 text-[#711b42]"><AlertCircle /><div><h2 className="font-black">Dashboard unavailable</h2><p className="mt-1 text-sm">{error}</p></div></Card>;

  const completed = analyses.filter((analysis) => analysis.status === "COMPLETED");
  const active = analyses.filter((analysis) => !["COMPLETED", "FAILED"].includes(analysis.status));
  const averageRisk = completed.length ? completed.reduce((sum, item) => sum + (item.change_risk_probability ?? 0), 0) / completed.length : null;

  return (
    <div className="space-y-10">
      <section className="grid gap-8 lg:grid-cols-[1fr_auto] lg:items-end">
        <div><p className="section-kicker">Engineering risk workspace</p><h1 className="mt-4 max-w-3xl text-4xl font-black leading-tight tracking-[-.04em] sm:text-6xl">Focus review effort <span className="ink-highlight">where it matters.</span></h1><p className="mt-5 max-w-2xl text-base leading-7 text-muted-ink">Transparent change risk, evidence-based file prioritization, and concrete testing recommendations.</p></div>
        <div className="rounded-2xl border-2 border-ink bg-mint p-4 font-mono text-xs font-bold shadow-[4px_4px_0_#191919]"><ShieldCheck className="mb-2" size={19} />Scores guide review.<br />They do not replace it.</div>
      </section>

      <section className="grid gap-5 sm:grid-cols-3" aria-label="Workspace summary">
        <Summary icon={<FolderGit2 />} label="Connected repositories" value={String(repositories.length)} tone="bg-sky" />
        <Summary icon={<Clock3 />} label="Completed analyses" value={String(completed.length)} tone="bg-lilac" />
        <Summary icon={<GitBranch />} label="Average change risk" value={percent(averageRisk)} tone="bg-butter" />
      </section>

      {active.length > 0 ? <Card className="overflow-hidden bg-butter"><div className="progress-stripes h-2 bg-[#d89020]" /><div className="flex items-center gap-3 p-5"><Loader2 className="animate-spin" size={20} /><div><p className="font-black">{active.length} analysis{active.length === 1 ? "" : "es"} in progress</p><p className="mt-1 text-sm text-muted-ink">This workspace updates when the analysis worker finishes.</p></div></div></Card> : null}

      <section>
        <div className="mb-5 flex flex-col justify-between gap-3 sm:flex-row sm:items-end"><div><p className="section-kicker">Your workspace</p><h2 className="mt-3 text-2xl font-black">Repositories</h2><p className="mt-1 text-sm text-muted-ink">Open a repository to inspect its latest analysis and ranked files.</p></div><span className="font-mono text-xs font-bold text-muted-ink">{repositories.length} CONNECTED</span></div>
        {repositories.length === 0 ? <Card className="dot-grid p-10 text-center"><FolderGit2 className="mx-auto" size={32} /><h3 className="mt-4 text-xl font-black">No repositories yet</h3><p className="mx-auto mt-2 max-w-md text-sm leading-6 text-muted-ink">Connect GitHub to choose a Python repository, or enable demo mode to explore a completed analysis.</p></Card> : <div className="grid gap-5 lg:grid-cols-2">{repositories.map((repository, index) => {
          const latest = analyses.find((analysis) => analysis.repository_id === repository.id);
          const tone = index % 2 === 0 ? "bg-paper" : "bg-[#fffaf0]";
          return <Link key={repository.id} href={`/repository/${repository.id}`} className="group"><Card className={`editorial-lift h-full p-6 ${tone}`}><div className="flex items-start justify-between gap-4"><div><p className="font-mono text-[10px] font-bold uppercase tracking-wider text-muted-ink">{repository.owner} / {repository.default_branch}</p><h3 className="mt-2 text-xl font-black">{repository.name}</h3></div><span className="grid size-10 place-items-center rounded-full border-2 border-ink bg-sky transition group-hover:bg-mint"><ArrowRight size={18} /></span></div><div className="mt-8 grid grid-cols-[1fr_auto] items-end gap-4 border-t-2 border-ink pt-5"><div><p className="font-mono text-[10px] font-bold uppercase tracking-wider text-muted-ink">Latest change risk</p><p className="mt-1 text-4xl font-black">{percent(latest?.change_risk_probability ?? null)}</p></div>{latest?.risk_level ? <RiskBadge level={latest.risk_level} /> : <span className="font-mono text-xs text-muted-ink">NOT ANALYZED</span>}</div></Card></Link>;
        })}</div>}
      </section>
    </div>
  );
}

function Summary({ icon, label, value, tone }: { icon: React.ReactNode; label: string; value: string; tone: string }) {
  return <Card className={`editorial-lift p-5 ${tone}`}><div className="flex items-center justify-between"><span className="grid size-10 place-items-center rounded-xl border-2 border-ink bg-paper">{icon}</span><span className="font-mono text-[10px] font-black uppercase tracking-wider text-muted-ink">Live</span></div><p className="mt-6 text-4xl font-black">{value}</p><p className="mt-1 text-sm font-semibold text-muted-ink">{label}</p></Card>;
}
