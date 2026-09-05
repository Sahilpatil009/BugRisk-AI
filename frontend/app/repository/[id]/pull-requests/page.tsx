import { AppShell } from "@/components/app-shell";
import { PullRequestsClient } from "@/components/pull-requests-client";

export default async function PullRequestsPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <AppShell><PullRequestsClient repositoryId={id} /></AppShell>;
}
