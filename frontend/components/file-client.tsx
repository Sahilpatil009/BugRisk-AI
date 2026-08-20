"use client";

import { ArrowLeft, CheckCircle2, Code2, GitCommit, GitPullRequestArrow, Network, ShieldCheck } from "lucide-react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { useEffect, useState } from "react";

import { RiskBadge } from "@/components/risk-badge";
import { Card, Skeleton } from "@/components/ui";
import { api, label, percent } from "@/lib/api";
import type { FileResult } from "@/lib/types";

const ContributionChart = dynamic(() => import("@/components/charts").then((module) => module.ContributionChart), { ssr: false, loading: () => <Skeleton className="mt-5 h-72" /> });

export function FileClient({ repositoryId, analysisId, fileId }: { repositoryId: string; analysisId: string; fileId: string }) {
  const [file, setFile] = useState<FileResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api<FileResult>(`/analyses/${analysisId}/files/${fileId}`).then(setFile).catch((reason: Error) => setError(reason.message)); }, [analysisId, fileId]);
  if (error) return <Card className="bg-pink p-6 text-[#711b42]">{error}</Card>;
  if (!file) return <div aria-label="Loading file evidence"><Skeleton className="h-24" /><Skeleton className="mt-6 h-[460px]" /></div>;

  return (
    <div className="space-y-8">
      <Link className="inline-flex items-center gap-2 font-mono text-xs font-bold underline decoration-2 underline-offset-4 hover:text-[#225b80]" href={`/repository/${repositoryId}`}><ArrowLeft size={15} />Back to repository</Link>
      <section className="flex flex-col justify-between gap-5 sm:flex-row sm:items-end"><div><p className="section-kicker">File review evidence</p><h1 className="mt-4 break-all font-mono text-2xl font-black tracking-tight sm:text-4xl">{file.file_path}</h1></div><RiskBadge level={file.risk_level} /></section>

      <section className="grid gap-5 lg:grid-cols-[.72fr_1.28fr]">
        <Card className="relative overflow-hidden bg-pink p-6"><ShieldCheck className="absolute right-5 top-5" /><p className="font-mono text-[10px] font-black uppercase tracking-widest text-[#711b42]">File-priority score</p><p className="mt-5 text-7xl font-black tracking-[-.06em]">{percent(file.file_priority_score)}</p><div className="mt-5 h-3 overflow-hidden rounded-full border-2 border-ink bg-paper"><div className="h-full border-r-2 border-ink bg-[#d86b91]" style={{ width: percent(file.file_priority_score) }} /></div><p className="mt-5 text-sm leading-7 text-[#711b42]">A review-ranking score derived from change risk and file evidence. It is not the probability that this file contains a defect.</p></Card>
        <Card className="p-6"><p className="section-kicker">Observed evidence</p><h2 className="mt-3 text-xl font-black">Code and history metrics</h2><div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-3"><Metric icon={<Code2 />} label="Lines of code" value={file.loc} tone="bg-sky" /><Metric icon={<GitPullRequestArrow />} label="Code churn" value={file.code_churn} tone="bg-butter" /><Metric icon={<GitCommit />} label="Commits" value={file.commit_count} tone="bg-lilac" /><Metric icon={<Network />} label="Dependencies" value={file.dependency_count} tone="bg-mint" /><Metric label="Complexity" value={file.complexity} tone="bg-peach" /><Metric label="Contributors" value={file.contributor_count} tone="bg-paper" /></div></Card>
      </section>

      <section className="grid gap-5 lg:grid-cols-2">
        <Card className="p-6"><p className="section-kicker">Model explanation</p><h2 className="mt-3 text-xl font-black">SHAP contribution</h2><p className="mt-2 text-sm text-muted-ink">Positive values increased the change-risk estimate; negative values reduced it.</p><ContributionChart explanations={file.explanations} /></Card>
        <Card className="overflow-hidden"><div className="border-b-2 border-ink bg-sand p-6"><p className="section-kicker">Feature detail</p><h2 className="mt-3 text-xl font-black">Why the model moved</h2></div><div className="divide-y-2 divide-ink/20">{file.explanations.length ? file.explanations.map((item) => <div key={item.feature_name} className="grid grid-cols-[1fr_auto] items-center gap-4 px-5 py-4"><div><p className="text-sm font-black">{label(item.feature_name)}</p><p className="mt-1 font-mono text-[10px] text-muted-ink">Observed value · {item.feature_value}</p></div><span className={`rounded-full border-2 border-ink px-2.5 py-1 font-mono text-xs font-black ${item.shap_value >= 0 ? "bg-pink text-[#711b42]" : "bg-mint text-[#185c38]"}`}>{item.shap_value >= 0 ? "+" : ""}{item.shap_value.toFixed(3)}</span></div>) : <p className="p-6 text-sm text-muted-ink">No explanation factors were saved for this file.</p>}</div></Card>
      </section>

      <Card className="overflow-hidden"><div className="border-b-2 border-ink bg-mint p-6"><p className="section-kicker">Next actions</p><h2 className="mt-3 text-xl font-black">Recommended review work</h2></div>{file.recommendations.length ? <div className="grid gap-4 p-6 md:grid-cols-3">{file.recommendations.map((recommendation, index) => <div key={recommendation} className={`flex gap-3 rounded-xl border-2 border-ink p-4 text-sm leading-6 ${index % 3 === 0 ? "bg-paper" : index % 3 === 1 ? "bg-butter" : "bg-sky"}`}><CheckCircle2 className="mt-0.5 shrink-0" size={18} /><span>{recommendation}</span></div>)}</div> : <p className="p-6 text-sm text-muted-ink">No recommendations were generated for this file.</p>}</Card>
    </div>
  );
}

function Metric({ icon, label: name, value, tone }: { icon?: React.ReactNode; label: string; value: number; tone: string }) {
  return <div className={`rounded-xl border-2 border-ink p-4 ${tone}`}><div className="flex items-center gap-2 font-mono text-[9px] font-black uppercase text-muted-ink">{icon}{name}</div><p className="mt-3 text-2xl font-black">{value}</p></div>;
}
