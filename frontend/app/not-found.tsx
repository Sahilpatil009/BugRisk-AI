import Link from "next/link";
import { Button } from "@/components/ui";

export default function NotFound() { return <main className="grid min-h-screen place-items-center px-5 text-center"><div className="editorial-card max-w-md rounded-2xl bg-butter p-8"><p className="font-mono text-xs font-black">ERROR 404</p><h1 className="mt-3 text-3xl font-black">Page not found</h1><p className="mt-3 text-sm text-muted-ink">The requested route is not part of this workspace.</p><Button asChild className="mt-6"><Link href="/dashboard">Return to dashboard</Link></Button></div></main>; }
