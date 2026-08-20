import { AppShell } from "@/components/app-shell";
import { RepositoryClient } from "@/components/repository-client";

export default async function RepositoryPage({ params }: { params: Promise<{ id: string }> }) { const { id } = await params; return <AppShell><RepositoryClient repositoryId={id} /></AppShell>; }

