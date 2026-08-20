import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "BugRisk AI", template: "%s · BugRisk AI" },
  description: "Prioritize code review and testing with explainable change-risk analysis.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}

