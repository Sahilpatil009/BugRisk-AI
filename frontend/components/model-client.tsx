"use client";

import { Activity, AlertCircle, CheckCircle2 } from "lucide-react";
import { useEffect, useState } from "react";

import { Card, Skeleton } from "@/components/ui";
import { api, label } from "@/lib/api";
import type { ModelMetrics } from "@/lib/types";

export function ModelClient() {
  const [model, setModel] = useState<ModelMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api<ModelMetrics>("/models/current/metrics").then(setModel).catch((reason: Error) => setError(reason.message)); }, []);
  if (error) return <Card className="p-6 text-rose-200">{error}</Card>;
  if (!model) return <Skeleton className="h-80" />;
  return <div className="space-y-7"><div><p className="text-sm font-semibold text-cyan-300">MODEL CARD</p><h1 className="mt-2 text-3xl font-bold">Current risk model</h1><p className="mt-2 max-w-2xl text-slate-400">Transparent performance status and the exact feature contract used by inference.</p></div><Card className="p-6"><div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-start"><div><p className="text-sm text-slate-500">{model.model_version}</p><h2 className="mt-1 text-2xl font-semibold">{label(model.model_name)}</h2></div><span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-sm ${model.trained ? "border-emerald-400/20 bg-emerald-400/10 text-emerald-200" : "border-amber-300/20 bg-amber-300/10 text-amber-200"}`}>{model.trained ? <CheckCircle2 size={15} /> : <AlertCircle size={15} />}{model.trained ? "Trained artifact" : "Demo fallback"}</span></div>{model.note && <p className="mt-5 rounded-xl border border-amber-300/15 bg-amber-300/[0.055] p-4 text-sm text-amber-100/80">{model.note}</p>}</Card><section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">{["precision", "recall", "f1", "roc_auc", "pr_auc"].map((metric) => <Card key={metric} className="p-5"><p className="text-xs uppercase tracking-wider text-slate-500">{label(metric)}</p><p className="mt-3 text-2xl font-bold">{model.metrics[metric] === undefined ? "—" : model.metrics[metric].toFixed(3)}</p></Card>)}</section><Card className="p-6"><h2 className="flex items-center gap-2 font-semibold"><Activity className="text-cyan-300" size={18} />Versioned feature schema</h2><div className="mt-5 flex flex-wrap gap-2">{model.feature_names.map((feature) => <span key={feature} className="rounded-lg border border-white/8 bg-white/[0.035] px-3 py-2 font-mono text-xs text-slate-300">{feature}</span>)}</div></Card></div>;
}

