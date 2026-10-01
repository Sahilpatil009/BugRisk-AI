"use client";

import { ExternalLink, GitPullRequest, Loader2, RefreshCw } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { RiskBadge } from "@/components/risk-badge";
import { Button, Card, Skeleton } from "@/components/ui";
import { api, percent } from "@/lib/api";
import type { PullRequest, Repository } from "@/lib/types";

export function PullRequestsClient({ repositoryId }: { repositoryId: string }) {
  const [repository, setRepository] = useState<Repository | null>(null);
  const [pullRequests, setPullRequests] = useState<PullRequest[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    const [repositories, items] = await Promise.all([
      api<Repository[]>("/repositories"),
      api<PullRequest[]>(`/repositories/${repositoryId}/pull-requests`),
    ]);
    const selected = repositories.find((item) => item.id === repositoryId);
    if (!selected) throw new Error("Repository not found");
    return { selected, items };
  }, [repositoryId]);

  const applyData = useCallback(({ selected, items }: { selected: Repository; items: PullRequest[] }) => {
    setRepository(selected);
    setPullRequests(items);
  }, []);

  useEffect(() => {
    void fetchData()
      .then(applyData)
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setLoading(false));
  }, [applyData, fetchData]);

  useEffect(() => {
    if (!pullRequests.some((item) => !["COMPLETED", "FAILED"].includes(item.status))) return;
    const timer = window.setInterval(
      () => void fetchData().then(applyData).catch(() => undefined),
      2500,
    );
    return () => window.clearInterval(timer);
  }, [applyData, fetchData, pullRequests]);

  if (loading) return <div aria-label="Loading pull requests"><Skeleton className="h-32" /><Skeleton className="mt-6 h-72" /></div>;
  if (!repository) return <Card className="bg-pink p-6 text-[#711b42]">{error ?? "Repository not found"}</Card>;

  return (
    <div className="space-y-8">
      <section className="flex flex-col justify-between gap-5 lg:flex-row lg:items-end">
        <div>
          <p className="section-kicker">{repository.owner} / {repository.name}</p>
          <h1 className="mt-4 text-4xl font-black tracking-[-.04em] sm:text-5xl">Pull request reports</h1>
          <p className="mt-4 max-w-2xl text-sm leading-7 text-muted-ink">Every opened or updated pull request is checked automatically. Only changed Python files enter the review-priority queue.</p>
        </div>
        <div className="flex flex-wrap gap-3">
          <Button asChild variant="secondary"><Link href={`/repository/${repository.id}`}>Repository overview</Link></Button>
          <Button onClick={() => void fetchData().then(applyData).catch((reason: Error) => setError(reason.message))}><RefreshCw className="mr-2" size={15} />Refresh</Button>
        </div>
      </section>

      {error ? <Card className="bg-pink p-4 text-sm text-[#711b42]">{error}</Card> : null}
      {pullRequests.length === 0 ? (
        <Card className="dot-grid p-10 text-center">
          <GitPullRequest className="mx-auto" size={34} />
          <h2 className="mt-4 text-xl font-black">No pull requests analyzed yet</h2>
          <p className="mx-auto mt-2 max-w-xl text-sm leading-6 text-muted-ink">Install the BugRisk AI GitHub App on this repository, then open or update a pull request. The signed webhook will create an analysis automatically.</p>
        </Card>
      ) : (
        <div className="grid gap-5">
          {pullRequests.map((item) => {
            const active = !["COMPLETED", "FAILED"].includes(item.status);
            return (
              <Card key={item.id} className="overflow-hidden">
                <div className="grid gap-5 p-6 md:grid-cols-[1fr_auto] md:items-center">
                  <div>
                    <div className="flex flex-wrap items-center gap-3">
                      <span className="rounded-full border-2 border-ink bg-mint px-3 py-1 font-mono text-[10px] font-black uppercase">PR #{item.github_pr_number}</span>
                      <span className="font-mono text-[10px] font-black uppercase text-muted-ink">{item.status}</span>
                      {active ? <Loader2 className="animate-spin" size={15} aria-label="Analysis in progress" /> : null}
                    </div>
                    <h2 className="mt-4 text-xl font-black">{item.title}</h2>
                    <p className="mt-2 font-mono text-xs text-muted-ink">{item.author} · {item.head_sha.slice(0, 10)} · {item.changed_files.length} changed files</p>
                  </div>
                  <div className="flex items-center gap-4 md:justify-end">
                    <div className="text-right"><p className="font-mono text-[9px] font-black uppercase text-muted-ink">Change risk</p><strong className="text-3xl font-black">{percent(item.risk_score)}</strong></div>
                    {item.risk_level ? <RiskBadge level={item.risk_level} /> : null}
                  </div>
                </div>
                <div className="flex flex-wrap gap-3 border-t-2 border-ink bg-sand px-6 py-4">
                  <Button asChild variant="secondary"><a href={item.html_url} target="_blank" rel="noreferrer">Open on GitHub <ExternalLink className="ml-2" size={14} /></a></Button>
                  {item.analysis_id ? <Button asChild variant="ghost"><Link href={`/repository/${repository.id}`}>View analysis</Link></Button> : null}
                </div>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
