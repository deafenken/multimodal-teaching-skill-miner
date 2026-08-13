import {Slot} from "@radix-ui/react-slot";
import {cva, type VariantProps} from "class-variance-authority";
import {forwardRef, type ButtonHTMLAttributes} from "react";

import {cn} from "@/lib/cn";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-colors disabled:pointer-events-none disabled:opacity-45",
  {
    variants: {
      variant: {
        default: "bg-[var(--app-text)] text-[var(--app-bg)] hover:opacity-90",
        subtle: "border border-transparent bg-transparent text-[var(--app-muted)] hover:border-[var(--app-border-strong)] hover:bg-[var(--app-hover)] hover:text-[var(--app-text)]",
        selected: "bg-[var(--app-selected)] text-[var(--app-text)] hover:bg-[var(--app-hover)]",
        drawer: "bg-[var(--app-surface-raised)] text-[var(--app-text)] shadow-sm hover:bg-[var(--app-hover)]"
      },
      size: {
        default: "h-9 px-3",
        sm: "h-7 px-2 text-xs",
        icon: "size-9 p-0"
      }
    },
    defaultVariants: {variant: "default", size: "default"}
  }
);

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & VariantProps<typeof buttonVariants> & {
  asChild?: boolean;
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button({className, variant, size, asChild = false, ...props}, ref) {
  const Component = asChild ? Slot : "button";
  return <Component ref={ref} className={cn(buttonVariants({variant, size}), className)} {...props} />;
});

Button.displayName = "Button";
