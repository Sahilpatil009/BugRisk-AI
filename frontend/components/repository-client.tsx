"use client";

import { AlertTriangle, ArrowDownUp, GitCommitHorizontal, Loader2, Play, RefreshCw } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import dynamic from "next/dynamic";
import { RiskBadge } from "@/components/risk-badge";
import { Button, Card, Skeleton } from "@/components/ui";
import { api, percent } from "@/lib/api";
import type { Analysis, FileResult, Repository } from "@/lib/types";

type Sort = "priority" | "complexity" | "churn" | "commits" | "name";

const RiskDistribution = dynamic(() => import("@/components/charts").then((module) => module.RiskDistribution), { ssr: false });

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
      setFiles([]);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Unable to start analysis"); }
  }

  if (loading) return <Skeleton className="h-96" />;
  if (!repository) return <Card className="p-6 text-rose-200">{error ?? "Repository not found"}</Card>;

  const counts = files.reduce<Record<string, number>>((result, file) => ({ ...result, [file.risk_level]: (result[file.risk_level] ?? 0) + 1 }), {});
  return (
    <div className="space-y-7">
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end"><div><p className="text-sm text-slate-500">{repository.owner} / repository</p><h1 className="mt-1 text-3xl font-bold">{repository.name}</h1><p className="mt-2 flex items-center gap-2 text-sm text-slate-400"><GitCommitHorizontal size={15} />{selected?.commit_sha?.slice(0, 10) ?? "No analyzed commit"}</p></div><Button onClick={startAnalysis} disabled={selected ? !["COMPLETED", "FAILED"].includes(selected.status) : false}>{selected && !["COMPLETED", "FAILED"].includes(selected.status) ? <Loader2 className="mr-2 animate-spin" size={16} /> : <Play className="mr-2" size={16} />}Analyze latest commit</Button></div>
      {error && <Card className="flex gap-3 border-rose-400/20 p-4 text-sm text-rose-200"><AlertTriangle size={18} />{error}</Card>}
      {selected && !["COMPLETED", "FAILED"].includes(selected.status) && <Card className="p-8 text-center"><Loader2 className="mx-auto animate-spin text-cyan-300" /><h2 className="mt-4 font-semibold">{selected.status.toLowerCase()} repository</h2><p className="mt-2 text-sm text-slate-500">This page updates automatically when the worker finishes.</p></Card>}
      {selected?.status === "FAILED" && <Card className="border-rose-400/20 p-6"><h2 className="font-semibold text-rose-200">Analysis failed</h2><p className="mt-2 text-sm text-slate-400">{selected.error_message}</p></Card>}
      {selected?.status === "COMPLETED" && <>
        <section className="grid gap-4 lg:grid-cols-[1.3fr_.7fr]">
          <Card className="p-6"><p className="text-xs font-semibold uppercase tracking-widest text-slate-500">Latest analysis</p><div className="mt-4 flex flex-wrap items-end gap-4"><span className="text-6xl font-black tracking-tight">{percent(selected.change_risk_probability)}</span>{selected.risk_level && <RiskBadge level={selected.risk_level} />}</div><p className="mt-4 max-w-lg text-sm leading-6 text-slate-400">This is the model&apos;s calibrated risk for the analyzed change. File scores below are prioritization indicators, not probabilities that a file contains a bug.</p><div className="mt-6 h-2 overflow-hidden rounded-full bg-white/5"><div className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-rose-400" style={{ width: percent(selected.change_risk_probability) }} /></div></Card>
          <Card className="p-4"><RiskDistribution files={files} /><div className="grid grid-cols-4 gap-1 text-center text-xs text-slate-400">{(["CRITICAL", "HIGH", "MEDIUM", "LOW"] as const).map((level) => <div key={level}><strong className="block text-base text-white">{counts[level] ?? 0}</strong>{level}</div>)}</div></Card>
        </section>
        <Card className="overflow-hidden"><div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/8 px-5 py-4"><div><h2 className="font-semibold">File review priority</h2><p className="mt-1 text-xs text-slate-500">Ranked from model change risk and repository evidence.</p></div><label className="flex items-center gap-2 text-sm text-slate-400"><ArrowDownUp size={15} /><span className="sr-only">Sort files</span><select value={sort} onChange={(event) => setSort(event.target.value as Sort)} className="rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-slate-200"><option value="priority">Priority</option><option value="complexity">Complexity</option><option value="churn">Churn</option><option value="commits">Commits</option><option value="name">File name</option></select></label></div><div className="overflow-x-auto"><table className="w-full min-w-[700px] text-left text-sm"><thead className="bg-white/[0.025] text-xs uppercase tracking-wider text-slate-500"><tr><th className="px-5 py-3">File</th><th className="px-5 py-3">Priority</th><th className="px-5 py-3">Complexity</th><th className="px-5 py-3">Churn</th><th className="px-5 py-3">Commits</th><th className="px-5 py-3">Level</th></tr></thead><tbody>{files.map((file) => <tr key={file.id} className="border-t border-white/6 hover:bg-white/[0.025]"><td className="px-5 py-4 font-mono text-xs text-cyan-200"><Link className="hover:underline" href={`/repository/${repository.id}/file/${file.id}?analysis=${selected.id}`}>{file.file_path}</Link></td><td className="px-5 py-4 font-bold">{percent(file.file_priority_score)}</td><td className="px-5 py-4 text-slate-400">{file.complexity}</td><td className="px-5 py-4 text-slate-400">{file.code_churn}</td><td className="px-5 py-4 text-slate-400">{file.commit_count}</td><td className="px-5 py-4"><RiskBadge level={file.risk_level} /></td></tr>)}</tbody></table></div></Card>
        <Card className="p-5"><div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-center"><div><h2 className="font-semibold">Analysis history</h2><p className="mt-1 text-sm text-slate-500">{analyses.length} saved run{analyses.length === 1 ? "" : "s"}</p></div><Button variant="secondary" onClick={() => { void fetchData().then(applyData).catch((reason: Error) => setError(reason.message)); }}><RefreshCw className="mr-2" size={15} />Refresh</Button></div></Card>
      </>}
    </div>
  );
}
