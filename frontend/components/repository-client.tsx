"use client";

import { AlertTriangle, ArrowDownUp, ArrowRight, GitCommitHorizontal, History, Loader2, Play, RefreshCw, ShieldCheck } from "lucide-react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { RiskBadge } from "@/components/risk-badge";
import { Button, Card, Skeleton } from "@/components/ui";
import { api, percent } from "@/lib/api";
import type { Analysis, FileResult, Repository } from "@/lib/types";

type Sort = "priority" | "complexity" | "churn" | "commits" | "name";

const RiskDistribution = dynamic(() => import("@/components/charts").then((module) => module.RiskDistribution), { ssr: false, loading: () => <Skeleton className="h-64" /> });

export function RepositoryClient({ repositoryId }: { repositoryId: string }) {
  const [repository, setRepository] = useState<Repository | null>(null);
  const [analyses, setAnalyses] = useState<Analysis[]>([]);
  const [selected, setSelected] = useState<Analysis | null>(null);
  const [files, setFiles] = useState<FileResult[]>([]);
  const [sort, setSort] = useState<Sort>("priority");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    const [repositories, analysisData] = await Promise.all([
      api<Repository[]>("/repositories"),
      api<Analysis[]>(`/analyses?repository_id=${repositoryId}`),
    ]);
    const repo = repositories.find((item) => item.id === repositoryId) ?? null;
    if (!repo) throw new Error("Repository not found");
    return { repo, analysisData };
  }, [repositoryId]);

  const applyData = useCallback(({ repo, analysisData }: { repo: Repository; analysisData: Analysis[] }) => {
    setRepository(repo);
    setAnalyses(analysisData);
    setSelected((current) => analysisData.find((item) => item.id === current?.id) ?? analysisData[0] ?? null);
  }, []);

  useEffect(() => {
    void fetchData().then(applyData).catch((reason: Error) => setError(reason.message)).finally(() => setLoading(false));
  }, [applyData, fetchData]);

  useEffect(() => {
    if (!selected || selected.status !== "COMPLETED") return;
    api<{ items: FileResult[] }>(`/analyses/${selected.id}/files?sort_by=${sort}&order=${sort === "name" ? "asc" : "desc"}`)
      .then((data) => setFiles(data.items)).catch((reason: Error) => setError(reason.message));
  }, [selected, sort]);

  useEffect(() => {
    if (!selected || ["COMPLETED", "FAILED"].includes(selected.status)) return;
    const timer = window.setInterval(() => { void fetchData().then(applyData).catch(() => undefined); }, 2000);
    return () => window.clearInterval(timer);
  }, [selected, applyData, fetchData]);

  async function startAnalysis() {
    if (!repository) return;
    setError(null);
    try {
      const created = await api<Analysis>("/analyses", { method: "POST", body: JSON.stringify({ repository_id: repository.id }) });
      setAnalyses((items) => [created, ...items]);
      setSelected(created);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Unable to start analysis"); }
  }

  if (loading) return <div aria-label="Loading repository"><Skeleton className="h-32" /><Skeleton className="mt-6 h-96" /></div>;
  if (!repository) return <Card className="bg-pink p-6 text-[#711b42]">{error ?? "Repository not found"}</Card>;

  const counts = files.reduce<Record<string, number>>((result, file) => ({ ...result, [file.risk_level]: (result[file.risk_level] ?? 0) + 1 }), {});
  const isActive = selected ? !["COMPLETED", "FAILED"].includes(selected.status) : false;

  return (
    <div className="space-y-8">
      <section className="flex flex-col justify-between gap-6 lg:flex-row lg:items-end">
        <div><p className="section-kicker">{repository.owner} / {repository.default_branch}</p><h1 className="mt-4 break-words text-4xl font-black tracking-[-.04em] sm:text-5xl">{repository.name}</h1><p className="mt-3 flex items-center gap-2 font-mono text-xs text-muted-ink"><GitCommitHorizontal size={15} />{selected?.commit_sha?.slice(0, 10) ?? "No analyzed commit"}</p></div>
        <Button onClick={startAnalysis} disabled={isActive} className="h-12 px-6">{isActive ? <Loader2 className="mr-2 animate-spin" size={16} /> : <Play className="mr-2" size={16} />}{isActive ? "Analysis running" : "Analyze latest commit"}</Button>
      </section>

      {error ? <Card className="flex gap-3 bg-pink p-4 text-sm text-[#711b42]"><AlertTriangle className="shrink-0" size={18} />{error}</Card> : null}
      {!selected ? <EmptyAnalysis /> : null}
      {selected && isActive ? <AnalysisProgress status={selected.status} /> : null}
      {selected?.status === "FAILED" ? <Card className="bg-pink p-6"><p className="font-mono text-xs font-black uppercase">Analysis failed</p><h2 className="mt-3 text-xl font-black">The worker could not complete this run.</h2><p className="mt-2 text-sm leading-6 text-[#711b42]">{selected.error_message}</p></Card> : null}

      {selected?.status === "COMPLETED" ? <>
        <section className="grid gap-5 lg:grid-cols-[1.25fr_.75fr]">
          <Card className="relative overflow-hidden bg-butter p-6 sm:p-8"><div className="absolute right-5 top-5 grid size-11 place-items-center rounded-full border-2 border-ink bg-paper"><ShieldCheck size={20} /></div><p className="font-mono text-[10px] font-black uppercase tracking-[.14em] text-muted-ink">Calibrated change risk</p><div className="mt-5 flex flex-wrap items-end gap-4"><span className="text-6xl font-black tracking-[-.06em] sm:text-7xl">{percent(selected.change_risk_probability)}</span>{selected.risk_level ? <RiskBadge level={selected.risk_level} /> : null}</div><p className="mt-5 max-w-xl text-sm leading-7 text-muted-ink">This is the model&apos;s estimate for the analyzed change. File scores rank review priority and are not probabilities that individual files contain defects.</p><div className="mt-7 h-3 overflow-hidden rounded-full border-2 border-ink bg-paper"><div className="h-full border-r-2 border-ink bg-[#e89aaa]" style={{ width: percent(selected.change_risk_probability) }} /></div></Card>
          <Card className="bg-paper p-4"><RiskDistribution files={files} /><div className="grid grid-cols-4 gap-1 text-center font-mono text-[9px] font-bold text-muted-ink">{(["CRITICAL", "HIGH", "MEDIUM", "LOW"] as const).map((level) => <div key={level}><strong className="block text-xl text-ink">{counts[level] ?? 0}</strong>{level}</div>)}</div></Card>
        </section>

        <Card className="overflow-hidden">
          <div className="flex flex-wrap items-center justify-between gap-4 border-b-2 border-ink bg-sand px-5 py-4"><div><p className="font-mono text-[10px] font-black uppercase tracking-wider text-muted-ink">Review queue</p><h2 className="mt-1 text-xl font-black">File priority</h2></div><label className="flex items-center gap-2 font-mono text-xs font-bold"><ArrowDownUp size={15} /><span className="sr-only">Sort files</span><select value={sort} onChange={(event) => setSort(event.target.value as Sort)} className="rounded-full border-2 border-ink bg-paper px-3 py-2"><option value="priority">Priority</option><option value="complexity">Complexity</option><option value="churn">Churn</option><option value="commits">Commits</option><option value="name">File name</option></select></label></div>
          {files.length === 0 ? <div className="p-10 text-center"><p className="font-black">No supported files were found.</p><p className="mt-2 text-sm text-muted-ink">The MVP currently extracts metrics from Python files.</p></div> : <div className="overflow-x-auto"><table className="w-full min-w-[760px] text-left text-sm"><thead className="bg-paper font-mono text-[10px] uppercase tracking-wider text-muted-ink"><tr><th className="px-5 py-3">File</th><th className="px-5 py-3">Priority</th><th className="px-5 py-3">Complexity</th><th className="px-5 py-3">Churn</th><th className="px-5 py-3">Commits</th><th className="px-5 py-3">Level</th><th className="px-5 py-3"><span className="sr-only">Open</span></th></tr></thead><tbody>{files.map((file) => <tr key={file.id} className="border-t-2 border-ink/20 transition hover:bg-sky/35"><td className="max-w-[300px] truncate px-5 py-4 font-mono text-xs font-bold"><Link className="underline decoration-2 underline-offset-4 hover:text-[#225b80]" href={`/repository/${repository.id}/file/${file.id}?analysis=${selected.id}`}>{file.file_path}</Link></td><td className="px-5 py-4 text-lg font-black">{percent(file.file_priority_score)}</td><td className="px-5 py-4 text-muted-ink">{file.complexity}</td><td className="px-5 py-4 text-muted-ink">{file.code_churn}</td><td className="px-5 py-4 text-muted-ink">{file.commit_count}</td><td className="px-5 py-4"><RiskBadge level={file.risk_level} /></td><td className="px-5 py-4"><ArrowRight size={16} /></td></tr>)}</tbody></table></div>}
        </Card>
      </> : null}

      {analyses.length > 0 ? <Card className="overflow-hidden"><div className="flex items-center justify-between border-b-2 border-ink bg-lilac px-5 py-4"><div><p className="font-mono text-[10px] font-black uppercase tracking-wider">Saved runs</p><h2 className="mt-1 text-xl font-black">Analysis history</h2></div><Button variant="secondary" onClick={() => { void fetchData().then(applyData).catch((reason: Error) => setError(reason.message)); }}><RefreshCw className="mr-2" size={14} />Refresh</Button></div><div className="divide-y-2 divide-ink/20">{analyses.map((analysis) => <button key={analysis.id} type="button" onClick={() => setSelected(analysis)} aria-pressed={selected?.id === analysis.id} className={`grid w-full grid-cols-[1fr_auto] items-center gap-4 px-5 py-4 text-left transition sm:grid-cols-[1fr_1fr_auto] ${selected?.id === analysis.id ? "bg-mint" : "bg-paper hover:bg-sand"}`}><div><p className="font-mono text-xs font-black">{analysis.commit_sha?.slice(0, 10) ?? "Pending commit"}</p><p className="mt-1 text-xs text-muted-ink">{new Date(analysis.created_at).toLocaleString()}</p></div><span className="hidden font-mono text-xs font-bold text-muted-ink sm:block">{analysis.status}</span><span className="flex items-center gap-3 font-black">{percent(analysis.change_risk_probability)}{selected?.id === analysis.id ? <ArrowRight size={16} /> : null}</span></button>)}</div></Card> : null}
    </div>
  );
}

function AnalysisProgress({ status }: { status: Analysis["status"] }) {
  const stages = ["QUEUED", "ANALYZING", "PREDICTING"] as const;
  const activeIndex = Math.max(0, stages.indexOf(status as (typeof stages)[number]));
  return <Card className="overflow-hidden bg-sky"><div className="progress-stripes h-2 bg-[#3c91c6]" /><div className="p-7"><Loader2 className="animate-spin" size={24} /><h2 className="mt-4 text-xl font-black">{status === "QUEUED" ? "Waiting for the analysis worker" : status === "ANALYZING" ? "Reading repository evidence" : "Generating risk predictions"}</h2><p className="mt-2 text-sm text-muted-ink">This page refreshes automatically. You can leave it open while the worker finishes.</p><ol className="mt-6 grid gap-3 sm:grid-cols-3">{stages.map((stage, index) => <li key={stage} className={`rounded-xl border-2 border-ink p-3 font-mono text-[10px] font-black ${index <= activeIndex ? "bg-paper" : "bg-paper/40"}`}>{index + 1}. {stage}</li>)}</ol></div></Card>;
}

function EmptyAnalysis() { return <Card className="dot-grid p-10 text-center"><History className="mx-auto" size={32} /><h2 className="mt-4 text-xl font-black">No analysis history</h2><p className="mx-auto mt-2 max-w-md text-sm leading-6 text-muted-ink">Analyze the latest commit to generate change risk, ranked files, evidence, and review recommendations.</p></Card>; }
