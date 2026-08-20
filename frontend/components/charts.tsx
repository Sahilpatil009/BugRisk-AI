"use client";

import { Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { Explanation, FileResult, RiskLevel } from "@/lib/types";
import { label } from "@/lib/api";

const COLORS: Record<RiskLevel, string> = { LOW: "#63bd91", MEDIUM: "#e7b43f", HIGH: "#e58b50", CRITICAL: "#d8668c" };

export function RiskDistribution({ files }: { files: FileResult[] }) {
  const data = (["CRITICAL", "HIGH", "MEDIUM", "LOW"] as RiskLevel[]).map((name) => ({ name, value: files.filter((file) => file.risk_level === name).length }));
  return (
    <div className="relative h-64" role="img" aria-label={`File priority distribution: ${data.map((item) => `${item.name.toLowerCase()} ${item.value}`).join(", ")}`}>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart><Pie data={data} dataKey="value" nameKey="name" innerRadius={58} outerRadius={86} paddingAngle={3} stroke="#191919" strokeWidth={2}>{data.map((entry) => <Cell key={entry.name} fill={COLORS[entry.name]} />)}</Pie><Tooltip contentStyle={{ background: "#fffdf8", color: "#191919", border: "2px solid #191919", borderRadius: 10, boxShadow: "3px 3px 0 #191919" }} /></PieChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ContributionChart({ explanations }: { explanations: Explanation[] }) {
  const data = explanations.map((item) => ({ ...item, name: label(item.feature_name) }));
  return (
    <div className="h-72" role="img" aria-label={`SHAP feature contributions: ${data.map((item) => `${item.name} ${item.shap_value.toFixed(3)}`).join(", ")}`}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 24 }}>
          <CartesianGrid stroke="#d9d1c1" horizontal={false} />
          <XAxis type="number" stroke="#625e56" tick={{ fontSize: 12 }} />
          <YAxis type="category" dataKey="name" width={118} stroke="#625e56" tick={{ fontSize: 11 }} />
          <Tooltip contentStyle={{ background: "#fffdf8", color: "#191919", border: "2px solid #191919", borderRadius: 10, boxShadow: "3px 3px 0 #191919" }} formatter={(value) => [Number(value).toFixed(3), "SHAP contribution"]} />
          <Bar dataKey="shap_value" radius={4} stroke="#191919" strokeWidth={1}>{data.map((entry) => <Cell key={entry.feature_name} fill={entry.shap_value >= 0 ? "#d8668c" : "#63bd91"} />)}</Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
