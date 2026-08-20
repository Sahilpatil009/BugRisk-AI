"use client";

import { AlertCircle, ArrowRight, Clock3, FolderGit2, Loader2 } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

import { api, percent } from "@/lib/api";
import type { Analysis, Repository } from "@/lib/types";
import { Card, Skeleton } from "@/components/ui";
import { RiskBadge } from "@/components/risk-badge";

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

  if (loading) return <div className="grid gap-4 md:grid-cols-3"><Skeleton className="h-40" /><Skeleton className="h-40" /><Skeleton className="h-40" /></div>;
  if (error) return <Card className="flex items-start gap-3 border-rose-400/20 p-6 text-rose-200"><AlertCircle /><div><h2 className="font-semibold">Dashboard unavailable</h2><p className="mt-1 text-sm text-rose-200/70">{error}</p></div></Card>;

  const completed = analyses.filter((analysis) => analysis.status === "COMPLETED");
  const active = analyses.filter((analysis) => !["COMPLETED", "FAILED"].includes(analysis.status));
  const averageRisk = completed.length ? completed.reduce((sum, item) => sum + (item.change_risk_probability ?? 0), 0) / completed.length : null;

  return (
    <div className="space-y-8">
      <div><p className="text-sm font-semibold text-cyan-300">ENGINEERING RISK WORKSPACE</p><h1 className="mt-2 text-3xl font-bold tracking-tight sm:text-4xl">Focus review effort where it matters.</h1><p className="mt-3 max-w-2xl text-slate-400">Model-based change risk, transparent file prioritization, and evidence-backed testing recommendations.</p></div>
      <section className="grid gap-4 sm:grid-cols-3" aria-label="Workspace summary">
        <Summary icon={<FolderGit2 />} label="Connected repositories" value={String(repositories.length)} />
        <Summary icon={<Clock3 />} label="Completed analyses" value={String(completed.length)} />
        <Summary icon={<Loader2 />} label="Average change risk" value={percent(averageRisk)} />
      </section>
      {active.length > 0 && <Card className="border-cyan-300/20 p-5"><p className="flex items-center gap-2 font-semibold text-cyan-200"><Loader2 className="animate-spin" size={18} />{active.length} analysis{active.length === 1 ? "" : "es"} in progress</p></Card>}
      <section>
        <div className="mb-4 flex items-end justify-between"><div><h2 className="text-xl font-semibold">Repositories</h2><p className="mt-1 text-sm text-slate-500">Select a repository to inspect its latest analysis.</p></div></div>
        <div className="grid gap-4 lg:grid-cols-2">
          {repositories.map((repository) => {
            const latest = analyses.find((analysis) => analysis.repository_id === repository.id);
            return <Link key={repository.id} href={`/repository/${repository.id}`} className="group"><Card className="h-full p-6 transition hover:-translate-y-0.5 hover:border-cyan-300/30"><div className="flex items-start justify-between gap-4"><div><p className="text-sm text-slate-500">{repository.owner}</p><h3 className="mt-1 text-lg font-semibold">{repository.name}</h3></div><ArrowRight className="text-slate-600 transition group-hover:translate-x-1 group-hover:text-cyan-300" /></div><div className="mt-7 flex items-end justify-between"><div><p className="text-xs uppercase tracking-wider text-slate-500">Latest change risk</p><p className="mt-1 text-2xl font-bold">{percent(latest?.change_risk_probability ?? null)}</p></div>{latest?.risk_level && <RiskBadge level={latest.risk_level} />}</div></Card></Link>;
          })}
        </div>
      </section>
    </div>
  );
}

function Summary({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return <Card className="p-5"><div className="flex items-center gap-3 text-slate-400"><span className="grid size-9 place-items-center rounded-lg bg-white/5 text-cyan-300">{icon}</span><span className="text-sm">{label}</span></div><p className="mt-4 text-3xl font-bold">{value}</p></Card>;
}

