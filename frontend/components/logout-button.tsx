"use client";

import { Github, Loader2, LogOut } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { api, authUrl } from "@/lib/api";

export function LogoutButton() {
  const router = useRouter();
  const [authenticated, setAuthenticated] = useState<boolean | null>(null);
  const [signingOut, setSigningOut] = useState(false);

  useEffect(() => {
    api<unknown[]>("/repositories")
      .then(() => setAuthenticated(true))
      .catch(() => setAuthenticated(false));
  }, []);

  async function signOut() {
    setSigningOut(true);
    try {
      await api<void>("/auth/logout", { method: "POST" });
      setAuthenticated(false);
      router.replace("/");
      router.refresh();
    } finally {
      setSigningOut(false);
    }
  }

  if (authenticated === null) return <span className="h-9 w-9" aria-hidden />;

  if (!authenticated) {
    return (
      <a href={authUrl} className="rounded-full px-3 py-2 hover:bg-sky focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-[#3c91c6]/30" aria-label="Connect GitHub">
        <Github className="mr-1.5 inline" size={14} />
        <span className="hidden md:inline">Connect GitHub</span>
      </a>
    );
  }

  return (
    <button
      type="button"
      onClick={signOut}
      disabled={signingOut}
      className="rounded-full px-3 py-2 hover:bg-pink focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-[#cf4a80]/35 disabled:opacity-60"
      aria-label="Sign out"
    >
      {signingOut ? <Loader2 className="mr-1.5 inline animate-spin" size={14} /> : <LogOut className="mr-1.5 inline" size={14} />}
      <span className="hidden md:inline">{signingOut ? "Signing out" : "Sign out"}</span>
    </button>
  );
}
