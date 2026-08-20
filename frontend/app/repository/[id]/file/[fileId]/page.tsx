import { AppShell } from "@/components/app-shell";
import { FileClient } from "@/components/file-client";

export default async function FilePage({ params, searchParams }: { params: Promise<{ id: string; fileId: string }>; searchParams: Promise<{ analysis?: string }> }) { const [{ id, fileId }, query] = await Promise.all([params, searchParams]); return <AppShell>{query.analysis ? <FileClient repositoryId={id} analysisId={query.analysis} fileId={fileId} /> : <p className="text-rose-200">Analysis identifier is required.</p>}</AppShell>; }

