import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import type { ButtonHTMLAttributes, HTMLAttributes } from "react";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex h-10 items-center justify-center rounded-full border-2 border-ink px-4 font-mono text-xs font-extrabold uppercase tracking-[.05em] transition-[transform,box-shadow,background-color] focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        primary: "bg-ink text-white shadow-[4px_4px_0_#cf4a80] hover:-translate-y-0.5 hover:shadow-[6px_6px_0_#cf4a80]",
        secondary: "bg-paper text-ink shadow-[3px_3px_0_#191919] hover:translate-x-0.5 hover:translate-y-0.5 hover:shadow-[1px_1px_0_#191919]",
        ghost: "border-transparent text-ink hover:border-ink hover:bg-butter",
      },
    },
    defaultVariants: { variant: "primary" },
  },
);

export function Button({ className, variant, asChild, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & VariantProps<typeof buttonVariants> & { asChild?: boolean }) {
  const Component = asChild ? Slot : "button";
  return <Component className={cn(buttonVariants({ variant }), className)} {...props} />;
}

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("editorial-card rounded-2xl", className)} {...props} />;
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded-xl border-2 border-ink/20 bg-sand", className)} aria-hidden />;
}
