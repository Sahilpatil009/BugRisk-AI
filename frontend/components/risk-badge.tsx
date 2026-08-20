import type { RiskLevel } from "@/lib/types";
import { cn } from "@/lib/utils";

const styles: Record<RiskLevel, string> = {
  LOW: "border-emerald-400/30 bg-emerald-400/10 text-emerald-300",
  MEDIUM: "border-amber-300/30 bg-amber-300/10 text-amber-200",
  HIGH: "border-orange-400/30 bg-orange-400/10 text-orange-300",
  CRITICAL: "border-rose-400/30 bg-rose-400/10 text-rose-300",
};

export function RiskBadge({ level }: { level: RiskLevel }) {
  return <span className={cn("inline-flex rounded-full border px-2.5 py-1 text-xs font-bold tracking-wide", styles[level])}>{level}</span>;
}

