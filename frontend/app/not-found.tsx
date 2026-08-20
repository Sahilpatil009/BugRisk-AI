import Link from "next/link";
import { Button } from "@/components/ui";

export default function NotFound() { return <main className="grid min-h-screen place-items-center bg-slate-950 text-center text-white"><div><p className="text-sm text-cyan-300">404</p><h1 className="mt-2 text-3xl font-bold">Page not found</h1><Button asChild className="mt-6"><Link href="/dashboard">Return to dashboard</Link></Button></div></main>; }

