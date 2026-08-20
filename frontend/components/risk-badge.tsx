import type { RiskLevel } from "@/lib/types";
import { cn } from "@/lib/utils";

const styles: Record<RiskLevel, string> = {
  LOW: "bg-mint text-[#185c38]",
  MEDIUM: "bg-butter text-[#6b4a00]",
  HIGH: "bg-peach text-[#7a3100]",
  CRITICAL: "bg-pink text-[#7b2048]",
};

export function RiskBadge({ level }: { level: RiskLevel }) {
  return <span className={cn("inline-flex rounded-full border-2 border-ink px-2.5 py-1 font-mono text-[11px] font-black tracking-wide shadow-[2px_2px_0_#191919]", styles[level])}>{level}</span>;
}
