"use client";

import { Button } from "@/components/ui";

export default function ErrorPage({ reset }: { error: Error & { digest?: string }; reset: () => void }) { return <main className="grid min-h-screen place-items-center bg-slate-950 px-5 text-center text-white"><div><h1 className="text-2xl font-bold">Something went wrong</h1><p className="mt-2 text-slate-400">The page could not be loaded.</p><Button className="mt-5" onClick={reset}>Try again</Button></div></main>; }

