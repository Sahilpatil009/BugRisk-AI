"use client";

import { ArrowLeft, CheckCircle2, Code2, GitCommit, GitPullRequestArrow, Network } from "lucide-react";
import Link from "next/link";
import dynamic from "next/dynamic";
import { useEffect, useState } from "react";

import { RiskBadge } from "@/components/risk-badge";
import { Card, Skeleton } from "@/components/ui";
import { api, label, percent } from "@/lib/api";
import type { FileResult } from "@/lib/types";

const ContributionChart = dynamic(() => import("@/components/charts").then((module) => module.ContributionChart), { ssr: false });

export function FileClient({ repositoryId, analysisId, fileId }: { repositoryId: string; analysisId: string; fileId: string }) {
  const [file, setFile] = useState<FileResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api<FileResult>(`/analyses/${analysisId}/files/${fileId}`).then(setFile).catch((reason: Error) => setError(reason.message)); }, [analysisId, fileId]);
  if (error) return <Card className="p-6 text-rose-200">{error}</Card>;
  if (!file) return <Skeleton className="h-[500px]" />;
  return (
    <div className="space-y-7">
      <Link className="inline-flex items-center gap-2 text-sm text-slate-400 hover:text-cyan-200" href={`/repository/${repositoryId}`}><ArrowLeft size={15} />Back to repository</Link>
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end"><div><p className="text-sm text-slate-500">File review evidence</p><h1 className="mt-2 break-all font-mono text-2xl font-bold text-cyan-100 sm:text-3xl">{file.file_path}</h1></div><RiskBadge level={file.risk_level} /></div>
      <section className="grid gap-4 lg:grid-cols-[.75fr_1.25fr]">
        <Card className="p-6"><p className="text-xs font-semibold uppercase tracking-widest text-slate-500">File-priority score</p><p className="mt-4 text-6xl font-black">{percent(file.file_priority_score)}</p><p className="mt-4 text-sm leading-6 text-slate-400">A review-ranking score derived from change risk and file evidence. It is not a probability that this file contains a defect.</p></Card>
        <Card className="p-6"><h2 className="font-semibold">Code and history metrics</h2><div className="mt-5 grid grid-cols-2 gap-4 sm:grid-cols-3"><Metric icon={<Code2 />} label="Lines of code" value={file.loc} /><Metric icon={<GitPullRequestArrow />} label="Code churn" value={file.code_churn} /><Metric icon={<GitCommit />} label="Commits" value={file.commit_count} /><Metric icon={<Network />} label="Dependencies" value={file.dependency_count} /><Metric label="Complexity" value={file.complexity} /><Metric label="Contributors" value={file.contributor_count} /></div></Card>
      </section>
      <section className="grid gap-4 lg:grid-cols-2"><Card className="p-6"><h2 className="font-semibold">SHAP contribution</h2><p className="mt-1 text-sm text-slate-500">Positive values increased the model&apos;s change-risk estimate.</p><ContributionChart explanations={file.explanations} /></Card><Card className="p-6"><h2 className="font-semibold">Evidence details</h2><div className="mt-5 space-y-3">{file.explanations.map((item) => <div key={item.feature_name} className="flex items-center justify-between rounded-xl bg-white/[0.035] px-4 py-3"><div><p className="text-sm font-medium">{label(item.feature_name)}</p><p className="mt-1 text-xs text-slate-500">Observed value: {item.feature_value}</p></div><span className={item.shap_value >= 0 ? "text-cyan-300" : "text-emerald-300"}>{item.shap_value >= 0 ? "+" : ""}{item.shap_value.toFixed(3)}</span></div>)}</div></Card></section>
      <Card className="p-6"><h2 className="font-semibold">Recommended review actions</h2><div className="mt-5 grid gap-3 md:grid-cols-3">{file.recommendations.map((recommendation) => <div key={recommendation} className="flex gap-3 rounded-xl border border-emerald-400/15 bg-emerald-400/[0.055] p-4 text-sm leading-6 text-slate-300"><CheckCircle2 className="mt-0.5 shrink-0 text-emerald-300" size={18} />{recommendation}</div>)}</div></Card>
    </div>
  );
}

function Metric({ icon, label: name, value }: { icon?: React.ReactNode; label: string; value: number }) {
  return <div className="rounded-xl bg-white/[0.035] p-4"><div className="flex items-center gap-2 text-xs text-slate-500">{icon}{name}</div><p className="mt-2 text-xl font-bold">{value}</p></div>;
}
