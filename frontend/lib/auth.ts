import { api } from "./api";

export interface AuthStatus {
  authenticated: boolean;
  github_connected: boolean;
}

export async function getAuthStatus(): Promise<AuthStatus> {
  return api<AuthStatus>("/auth/status");
}

export function githubAuthUrl(): string {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
  return `${apiUrl}/auth/github`;
}
