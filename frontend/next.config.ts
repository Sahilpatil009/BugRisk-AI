import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  agentRules: false,
  // Vercel injects a Next.js build adapter that currently conflicts with
  // standalone output on Next 16.3. Keep standalone bundles for Docker while
  // letting Vercel package the application with its native adapter.
  output: process.env.VERCEL ? undefined : "standalone",
};

export default nextConfig;
