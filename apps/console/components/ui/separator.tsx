import * as SeparatorPrimitive from "@radix-ui/react-separator";

import {cn} from "@/lib/cn";

export function Separator({className, orientation = "horizontal"}: {className?: string; orientation?: "horizontal" | "vertical"}) {
  return (
    <SeparatorPrimitive.Root
      decorative
      orientation={orientation}
      className={cn(
        "shrink-0 bg-[var(--border)]",
        orientation === "horizontal" ? "h-px w-full" : "h-full w-px",
        className
      )}
    />
  );
}
