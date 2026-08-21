import {
  ArrowRight,
  BarChart3,
  CheckCircle2,
  GitBranch,
  Github,
  History,
  ScanSearch,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import Link from "next/link";
import type { ReactNode } from "react";

import { Button, Card } from "@/components/ui";
import { authUrl } from "@/lib/api";

const workflow = [
  { number: "01", title: "Connect", text: "Choose a Python repository through GitHub. Tokens remain encrypted on the server.", icon: Github, tone: "bg-sky" },
  { number: "02", title: "Analyze", text: "The worker reads change history, churn, complexity, ownership, and defect signals.", icon: ScanSearch, tone: "bg-butter" },
  { number: "03", title: "Prioritize", text: "Review the highest-priority files first, with evidence and recommended actions.", icon: ShieldCheck, tone: "bg-mint" },
];

const faqs = [
  ["Does a high file score mean the file has a bug?", "No. File priority is a review-ranking signal derived from change risk and file evidence. It is not the probability that a file contains a defect."],
  ["What does the model predict?", "The supervised model estimates commit or change risk. That matches the commit-level labels used during training."],
  ["Can I understand why a score is high?", "Yes. SHAP contributions show which observed signals increased or reduced the estimate, and recommendations cite those signals."],
  ["Which repositories are supported?", "The MVP analyzes Python repositories. Support for more languages is planned after the baseline is validated."],
];

export default function Home() {
  return (
    <main id="top" className="mx-auto max-w-7xl px-5 pb-10 text-ink sm:px-7">
      <header className="flex h-20 items-center justify-between">
        <Link href="#top" className="flex items-center gap-2.5 font-mono text-sm font-black" aria-label="BugRisk AI home">
          <span className="grid size-10 place-items-center rounded-xl border-2 border-ink bg-pink shadow-[3px_3px_0_#191919]"><ScanSearch size={19} /></span>
          BUGRISK<span className="-ml-2.5 text-[#a13363]">.AI</span>
        </Link>
        <nav className="hidden items-center gap-7 font-mono text-xs font-bold md:flex" aria-label="Landing navigation">
          <Link href="#workflow" className="hover:underline">How it works</Link>
          <Link href="#scores" className="hover:underline">Scores</Link>
          <Link href="#faq" className="hover:underline">FAQ</Link>
        </nav>
        <Button asChild className="h-10 px-4"><a href={authUrl}><Github className="mr-2" size={15} />Sign in</a></Button>
      </header>

      <section className="page-enter grid gap-12 py-14 lg:grid-cols-[.95fr_1.05fr] lg:items-center lg:py-20">
        <div>
          <p className="section-kicker">Explainable engineering risk</p>
          <h1 className="mt-6 max-w-[10ch] text-5xl font-black leading-[.94] tracking-[-.055em] sm:text-7xl lg:text-[5.7rem]">
            Review the right code <span className="ink-highlight">first.</span>
          </h1>
          <p className="mt-7 max-w-xl text-lg leading-8 text-muted-ink">
            BugRisk AI turns repository history and code metrics into transparent change-risk estimates, file priorities, and focused review actions.
          </p>
          <div className="mt-9 flex flex-wrap gap-4">
            <Button asChild className="h-12 px-6"><a href={authUrl}><Github className="mr-2" size={17} />Continue with GitHub</a></Button>
            <Button asChild variant="secondary" className="h-12 px-6"><Link href="/dashboard">Explore demo <ArrowRight className="ml-2" size={17} /></Link></Button>
          </div>
          <div className="mt-8 flex flex-wrap gap-x-6 gap-y-2 font-mono text-[11px] font-bold text-muted-ink">
            <span className="flex items-center gap-1.5"><CheckCircle2 size={14} />Evidence with every score</span>
            <span className="flex items-center gap-1.5"><ShieldCheck size={14} />Human review stays in control</span>
          </div>
        </div>
        <AnalysisPreview />
      </section>

      <section className="grid border-y-2 border-ink py-5 font-mono text-xs font-bold uppercase tracking-wider sm:grid-cols-[auto_1fr] sm:items-center sm:gap-10">
        <span className="text-muted-ink">Built for practical review</span>
        <div className="mt-4 flex flex-wrap justify-between gap-4 sm:mt-0"><span>Git history</span><span>Code metrics</span><span>SHAP evidence</span><span>Deterministic actions</span></div>
      </section>

      <section id="workflow" className="py-24">
        <p className="section-kicker">One focused workflow</p>
        <h2 className="mt-4 max-w-3xl text-4xl font-black leading-tight tracking-[-.04em] sm:text-6xl">From repository to a <span className="ink-highlight mint">review plan.</span></h2>
        <div className="mt-12 grid gap-6 md:grid-cols-3">
          {workflow.map((step) => <Card key={step.number} className="editorial-lift p-6"><span className="font-mono text-xs font-black text-muted-ink">{step.number}</span><span className={`mt-5 grid size-12 place-items-center rounded-full border-2 border-ink shadow-[3px_3px_0_#191919] ${step.tone}`}><step.icon size={20} /></span><h3 className="mt-6 text-xl font-black">{step.title}</h3><p className="mt-3 text-sm leading-7 text-muted-ink">{step.text}</p></Card>)}
        </div>
      </section>

      <section id="scores" className="grid gap-10 rounded-3xl border-2 border-ink bg-paper p-6 shadow-[8px_8px_0_#191919] lg:grid-cols-[.8fr_1.2fr] lg:p-10">
        <div><p className="section-kicker">Two scores, two meanings</p><h2 className="mt-4 text-4xl font-black leading-tight tracking-[-.04em] sm:text-5xl">Clear language builds better decisions.</h2><p className="mt-5 text-sm leading-7 text-muted-ink">The model predicts risk for an analyzed change. File priority then ranks where review effort could be most valuable. BugRisk AI never relabels that ranking as a bug probability.</p></div>
        <div className="grid gap-4 sm:grid-cols-2">
          <ScoreCard tone="bg-sky" label="Change-risk probability" value="78%" text="A calibrated estimate for the analyzed commit or change." icon={<GitBranch size={20} />} />
          <ScoreCard tone="bg-lilac" label="File-priority score" value="91" text="A ranking signal combining change risk with file evidence." icon={<BarChart3 size={20} />} />
          <div className="rounded-2xl border-2 border-ink bg-mint p-5"><b className="font-mono text-xs">LOW · 0–30</b><p className="mt-2 text-sm text-[#185c38]">Normal review attention</p></div>
          <div className="rounded-2xl border-2 border-ink bg-butter p-5"><b className="font-mono text-xs">MEDIUM · &gt;30–60</b><p className="mt-2 text-sm text-[#6b4a00]">Check important signals</p></div>
          <div className="rounded-2xl border-2 border-ink bg-peach p-5"><b className="font-mono text-xs">HIGH · &gt;60–80</b><p className="mt-2 text-sm text-[#7a3100]">Prioritize deeper review</p></div>
          <div className="rounded-2xl border-2 border-ink bg-pink p-5"><b className="font-mono text-xs">CRITICAL · &gt;80–100</b><p className="mt-2 text-sm text-[#7b2048]">Review and test first</p></div>
        </div>
      </section>

      <section className="grid gap-12 py-24 lg:grid-cols-[.8fr_1.2fr] lg:items-start">
        <div><p className="section-kicker">Evidence, not mystery</p><h2 className="mt-4 text-4xl font-black leading-tight tracking-[-.04em] sm:text-5xl">Every recommendation has a reason.</h2><p className="mt-5 text-sm leading-7 text-muted-ink">Dominant SHAP factors and observed code metrics become concrete review and testing suggestions. An optional LLM may rewrite the wording, but it never changes a score.</p></div>
        <div className="space-y-4">
          <Evidence index="01" tone="bg-pink" title="High recent churn" text="Review changed branches and add regression coverage around the modified paths." />
          <Evidence index="02" tone="bg-butter" title="Rising complexity" text="Split dense functions and test conditional paths before merging the change." />
          <Evidence index="03" tone="bg-mint" title="Broad ownership" text="Request review from a maintainer familiar with the affected subsystem." />
        </div>
      </section>

      <section className="grid gap-8 rounded-3xl border-2 border-ink bg-lilac p-7 shadow-[8px_8px_0_#191919] lg:grid-cols-[1fr_auto] lg:items-end lg:p-12">
        <div><p className="section-kicker">Transparent by design</p><h2 className="mt-4 max-w-3xl text-4xl font-black leading-tight tracking-[-.04em] sm:text-5xl">See the model before you trust the result.</h2><p className="mt-5 max-w-2xl text-sm leading-7 text-muted-ink">The model card exposes artifact status, versioned features, precision, recall, F1, ROC-AUC, and PR-AUC.</p></div>
        <Button asChild variant="secondary" className="h-12 px-6"><Link href="/models">Open model card <ArrowRight className="ml-2" size={17} /></Link></Button>
      </section>

      <section id="faq" className="py-24" aria-labelledby="faq-title">
        <p className="section-kicker">Questions, answered</p><h2 id="faq-title" className="mt-4 text-4xl font-black tracking-[-.04em] sm:text-5xl">Know what the score can—and cannot—say.</h2>
        <div className="mt-10 divide-y-2 divide-ink border-y-2 border-ink">{faqs.map(([question, answer]) => <details key={question} className="group py-1"><summary className="flex cursor-pointer list-none items-center justify-between gap-5 py-5 font-bold"><span>{question}</span><span className="grid size-8 shrink-0 place-items-center rounded-full border-2 border-ink bg-paper font-mono transition group-open:rotate-45">+</span></summary><p className="max-w-3xl pb-6 text-sm leading-7 text-muted-ink">{answer}</p></details>)}</div>
      </section>

      <section className="rounded-3xl border-2 border-ink bg-mint p-8 text-center shadow-[8px_8px_0_#191919] sm:p-12"><Sparkles className="mx-auto" /><h2 className="mx-auto mt-5 max-w-3xl text-4xl font-black tracking-[-.04em] sm:text-5xl">Spend review time where the evidence points.</h2><div className="mt-8 flex flex-wrap justify-center gap-4"><Button asChild className="h-12 px-6"><a href={authUrl}>Connect GitHub</a></Button><Button asChild variant="secondary" className="h-12 px-6"><Link href="/dashboard">Open demo</Link></Button></div></section>

      <footer className="mt-20 flex flex-col gap-5 border-t-2 border-ink py-8 font-mono text-xs text-muted-ink sm:flex-row sm:items-center sm:justify-between"><span className="font-black text-ink">BUGRISK.AI</span><span>Explainable review prioritization for Python repositories.</span><nav className="flex gap-5"><Link href="/dashboard" className="hover:text-ink">Dashboard</Link><Link href="/models" className="hover:text-ink">Model card</Link></nav></footer>
    </main>
  );
}

function AnalysisPreview() {
  return <div className="relative mx-auto w-full max-w-2xl lg:pl-6"><div className="soft-float absolute -right-2 -top-5 z-10 rounded-xl border-2 border-ink bg-butter px-4 py-3 font-mono text-[10px] font-black shadow-[3px_3px_0_#191919]">CHANGE RISK · 78%</div><Card className="overflow-hidden shadow-[10px_10px_0_#191919]"><div className="flex items-center justify-between border-b-2 border-ink bg-sand px-4 py-3 font-mono text-[10px] font-bold"><span className="flex gap-1.5"><i className="size-2 rounded-full bg-pink" /><i className="size-2 rounded-full bg-butter" /><i className="size-2 rounded-full bg-mint" /></span><span>ecommerce-backend / analysis</span><span className="rounded-full border border-ink bg-mint px-2 py-0.5">complete</span></div><div className="grid min-h-[390px] sm:grid-cols-[110px_1fr]"><aside className="hidden border-r-2 border-ink bg-[#f1eadb] p-3 font-mono text-[9px] font-bold text-muted-ink sm:block"><b className="mb-7 grid size-9 place-items-center rounded-lg border-2 border-ink bg-pink text-ink">BR</b><p className="rounded-md bg-ink p-2 text-white">Overview</p><p className="p-2">Files</p><p className="p-2">Evidence</p><p className="p-2">History</p></aside><div className="p-5 sm:p-6"><div className="flex items-start justify-between gap-4"><div><p className="font-mono text-[9px] font-bold uppercase tracking-widest text-muted-ink">Latest analysis</p><h2 className="mt-1 text-xl font-black">Review queue</h2></div><History size={18} /></div><div className="mt-5 grid gap-3 sm:grid-cols-3"><MiniStat label="Change risk" value="78%" tone="bg-butter" /><MiniStat label="Files ranked" value="24" tone="bg-sky" /><MiniStat label="Critical" value="3" tone="bg-pink" /></div><div className="mt-6 overflow-hidden rounded-xl border-2 border-ink"><div className="grid grid-cols-[1fr_auto] border-b-2 border-ink bg-sand px-3 py-2 font-mono text-[9px] font-black"><span>FILE</span><span>PRIORITY</span></div><PreviewFile name="services/payment.py" score="94" tone="bg-pink" /><PreviewFile name="api/checkout.py" score="82" tone="bg-pink" /><PreviewFile name="models/order.py" score="71" tone="bg-peach" /><PreviewFile name="utils/currency.py" score="38" tone="bg-butter" /></div><p className="mt-5 flex items-center gap-2 text-xs text-muted-ink"><ShieldCheck size={15} />Ranked for review—not labelled as defective.</p></div></div></Card></div>;
}

function MiniStat({ label, value, tone }: { label: string; value: string; tone: string }) { return <div className={`rounded-xl border-2 border-ink p-3 ${tone}`}><p className="font-mono text-[8px] font-black uppercase">{label}</p><strong className="mt-2 block text-xl">{value}</strong></div>; }
function PreviewFile({ name, score, tone }: { name: string; score: string; tone: string }) { return <div className="grid grid-cols-[1fr_auto] items-center border-b border-ink/20 px-3 py-3 text-xs last:border-0"><span className="truncate font-mono">{name}</span><b className={`rounded-full border border-ink px-2 py-0.5 font-mono text-[10px] ${tone}`}>{score}</b></div>; }
function ScoreCard({ tone, label, value, text, icon }: { tone: string; label: string; value: string; text: string; icon: ReactNode }) { return <div className={`rounded-2xl border-2 border-ink p-5 shadow-[3px_3px_0_#191919] ${tone}`}><span className="grid size-9 place-items-center rounded-full border-2 border-ink bg-paper">{icon}</span><p className="mt-5 font-mono text-[10px] font-black uppercase tracking-wider">{label}</p><strong className="mt-2 block text-4xl">{value}</strong><p className="mt-3 text-sm leading-6 text-muted-ink">{text}</p></div>; }
function Evidence({ index, tone, title, text }: { index: string; tone: string; title: string; text: string }) { return <div className="grid grid-cols-[auto_1fr] gap-4 border-b-2 border-ink pb-4"><span className={`grid size-11 place-items-center rounded-xl border-2 border-ink font-mono text-xs font-black ${tone}`}>{index}</span><div><h3 className="font-black">{title}</h3><p className="mt-1 text-sm leading-6 text-muted-ink">{text}</p></div></div>; }
