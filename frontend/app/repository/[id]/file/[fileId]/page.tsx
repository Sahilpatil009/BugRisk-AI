import { AppShell } from "@/components/app-shell";
import { FileClient } from "@/components/file-client";

export default async function FilePage({ params, searchParams }: { params: Promise<{ id: string; fileId: string }>; searchParams: Promise<{ analysis?: string }> }) { const [{ id, fileId }, query] = await Promise.all([params, searchParams]); return <AppShell>{query.analysis ? <FileClient repositoryId={id} analysisId={query.analysis} fileId={fileId} /> : <div className="editorial-card rounded-2xl bg-pink p-6 text-[#711b42]">Analysis identifier is required.</div>}</AppShell>; }
