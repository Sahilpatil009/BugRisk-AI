"use client";

import { Activity, AlertCircle, CheckCircle2, Database, ShieldCheck } from "lucide-react";
import { useEffect, useState } from "react";

import { Card, Skeleton } from "@/components/ui";
import { api, label } from "@/lib/api";
import type { ModelMetrics } from "@/lib/types";

const metricDescriptions: Record<string, string> = {
  precision: "How often flagged changes were defective",
  recall: "How many defective changes were identified",
  f1: "Balance between precision and recall",
  roc_auc: "Ranking quality across thresholds",
  pr_auc: "Performance on the imbalanced target",
};

export function ModelClient() {
  const [model, setModel] = useState<ModelMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api<ModelMetrics>("/models/current/metrics").then(setModel).catch((reason: Error) => setError(reason.message)); }, []);
  if (error) return <Card className="bg-pink p-6 text-[#711b42]">{error}</Card>;
  if (!model) return <div aria-label="Loading model card"><Skeleton className="h-40" /><Skeleton className="mt-6 h-80" /></div>;

  return (
    <div className="space-y-8">
      <section><p className="section-kicker">Model transparency</p><h1 className="mt-4 max-w-3xl text-4xl font-black tracking-[-.04em] sm:text-6xl">Know the model behind <span className="ink-highlight mint">the score.</span></h1><p className="mt-5 max-w-2xl text-base leading-7 text-muted-ink">Artifact status, evaluation metrics, and the versioned feature contract used during inference.</p></section>

      <Card className={`p-6 sm:p-8 ${model.trained ? "bg-mint" : "bg-butter"}`}><div className="flex flex-col justify-between gap-5 sm:flex-row sm:items-start"><div><p className="font-mono text-[10px] font-black uppercase tracking-widest text-muted-ink">{model.model_version}</p><h2 className="mt-2 text-3xl font-black">{label(model.model_name)}</h2></div><span className="inline-flex w-fit items-center gap-2 rounded-full border-2 border-ink bg-paper px-3 py-2 font-mono text-[10px] font-black shadow-[2px_2px_0_#191919]">{model.trained ? <CheckCircle2 size={15} /> : <AlertCircle size={15} />}{model.trained ? "TRAINED ARTIFACT" : "DEMO FALLBACK"}</span></div>{model.note ? <p className="mt-6 max-w-3xl rounded-xl border-2 border-ink bg-paper/70 p-4 text-sm leading-6 text-muted-ink">{model.note}</p> : null}</Card>

      <section><div className="mb-5"><p className="section-kicker">Evaluation</p><h2 className="mt-3 text-2xl font-black">Performance metrics</h2></div><div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">{["precision", "recall", "f1", "roc_auc", "pr_auc"].map((metric, index) => <Card key={metric} className={`editorial-lift p-5 ${["bg-sky", "bg-pink", "bg-lilac", "bg-butter", "bg-mint"][index]}`}><p className="font-mono text-[10px] font-black uppercase tracking-wider">{label(metric)}</p><p className="mt-4 text-4xl font-black">{model.metrics[metric] === undefined ? "—" : model.metrics[metric].toFixed(3)}</p><p className="mt-3 text-xs leading-5 text-muted-ink">{metricDescriptions[metric]}</p></Card>)}</div></section>

      <section className="grid gap-5 lg:grid-cols-[1.25fr_.75fr]">
        <Card className="p-6"><h2 className="flex items-center gap-2 text-xl font-black"><Activity size={20} />Versioned feature schema</h2><p className="mt-2 text-sm leading-6 text-muted-ink">Inference must supply these features in the exact order used during training.</p><div className="mt-6 flex flex-wrap gap-2">{model.feature_names.map((feature, index) => <span key={feature} className={`rounded-lg border-2 border-ink px-3 py-2 font-mono text-[11px] font-bold ${index % 4 === 0 ? "bg-sky" : index % 4 === 1 ? "bg-butter" : index % 4 === 2 ? "bg-mint" : "bg-lilac"}`}>{feature}</span>)}</div></Card>
        <Card className="bg-paper p-6"><ShieldCheck size={24} /><h2 className="mt-4 text-xl font-black">Important limitations</h2><ul className="mt-4 space-y-3 text-sm leading-6 text-muted-ink"><li>• Predicts commit/change risk, not guaranteed defects.</li><li>• File priority is a derived review-ranking score.</li><li>• Initial analysis scope is Python repositories.</li><li>• Human review remains the final decision point.</li></ul><div className="mt-6 flex items-center gap-2 border-t-2 border-ink pt-5 font-mono text-[10px] font-black"><Database size={15} />APACHEJIT-ALIGNED BASELINE</div></Card>
      </section>
    </div>
  );
}
