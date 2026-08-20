import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "BugRisk AI", template: "%s · BugRisk AI" },
  description: "Prioritize code review and testing with transparent change-risk analysis and evidence-backed file ranking.",
  applicationName: "BugRisk AI",
  keywords: ["software quality", "code review", "risk analysis", "GitHub", "explainable AI"],
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body className="antialiased">{children}</body></html>;
}
