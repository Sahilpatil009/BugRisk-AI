"use client";

import { Button } from "@/components/ui";

export default function ErrorPage({ reset }: { error: Error & { digest?: string }; reset: () => void }) { return <main className="grid min-h-screen place-items-center px-5 text-center"><div className="editorial-card max-w-md rounded-2xl bg-pink p-8"><p className="font-mono text-xs font-black">APPLICATION ERROR</p><h1 className="mt-3 text-3xl font-black">Something went wrong</h1><p className="mt-3 text-sm text-[#711b42]">The page could not be loaded.</p><Button className="mt-6" onClick={reset}>Try again</Button></div></main>; }
