"use client";

import { Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { Explanation, FileResult, RiskLevel } from "@/lib/types";
import { label } from "@/lib/api";

const COLORS: Record<RiskLevel, string> = { LOW: "#34d399", MEDIUM: "#fcd34d", HIGH: "#fb923c", CRITICAL: "#fb7185" };

export function RiskDistribution({ files }: { files: FileResult[] }) {
  const data = (["CRITICAL", "HIGH", "MEDIUM", "LOW"] as RiskLevel[]).map((name) => ({ name, value: files.filter((file) => file.risk_level === name).length }));
  return (
    <div className="h-64" aria-label="File priority distribution chart">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart><Pie data={data} dataKey="value" nameKey="name" innerRadius={58} outerRadius={86} paddingAngle={4}>{data.map((entry) => <Cell key={entry.name} fill={COLORS[entry.name]} />)}</Pie><Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 12 }} /></PieChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ContributionChart({ explanations }: { explanations: Explanation[] }) {
  const data = explanations.map((item) => ({ ...item, name: label(item.feature_name) }));
  return (
    <div className="h-72" aria-label="SHAP feature contribution chart">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 24 }}>
          <CartesianGrid stroke="#1e293b" horizontal={false} />
          <XAxis type="number" stroke="#64748b" tick={{ fontSize: 12 }} />
          <YAxis type="category" dataKey="name" width={118} stroke="#94a3b8" tick={{ fontSize: 11 }} />
          <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 12 }} formatter={(value) => [Number(value).toFixed(3), "SHAP contribution"]} />
          <Bar dataKey="shap_value" radius={6}>{data.map((entry) => <Cell key={entry.feature_name} fill={entry.shap_value >= 0 ? "#22d3ee" : "#34d399"} />)}</Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

