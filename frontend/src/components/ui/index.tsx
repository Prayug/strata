/* Dense, terminal-style primitives built on Radix (shadcn/ui pattern, restyled). */
import * as DialogP from "@radix-ui/react-dialog";
import * as DropdownP from "@radix-ui/react-dropdown-menu";
import * as PopoverP from "@radix-ui/react-popover";
import * as SwitchP from "@radix-ui/react-switch";
import * as TabsP from "@radix-ui/react-tabs";
import * as ToggleGroupP from "@radix-ui/react-toggle-group";
import * as TooltipP from "@radix-ui/react-tooltip";
import { cva, type VariantProps } from "class-variance-authority";
import { X } from "lucide-react";
import * as React from "react";
import { cn } from "@/lib/format";

const button = cva(
  "inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-sm text-xs font-medium transition-colors select-none disabled:opacity-40 disabled:pointer-events-none [&_svg]:size-3.5 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "bg-bg-3 text-fg border border-line-2 hover:bg-bg-4 hover:border-faint",
        primary: "bg-accent text-[#1a1206] hover:bg-accent-2 font-semibold",
        ghost: "text-fg-2 hover:bg-bg-3 hover:text-fg",
        outline: "border border-line-2 text-fg-2 hover:text-fg hover:border-faint",
        danger: "text-down border border-down/40 hover:bg-down/10",
      },
      size: { sm: "h-6 px-2", md: "h-7 px-2.5", lg: "h-8 px-3 text-sm", icon: "h-6 w-6", iconSm: "h-5 w-5" },
    },
    defaultVariants: { variant: "default", size: "md" },
  },
);
export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof button> {}
export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant, size, ...p }, ref) => (
  <button ref={ref} className={cn(button({ variant, size }), className)} {...p} />
));

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement> & { sz?: "sm" | "md" }>(({ className, sz = "md", ...p }, ref) => (
  <input ref={ref} className={cn("field", sz === "sm" && "field-sm", p.type === "number" && "num", className)} {...p} />
));

/** Number input that tolerates empty / partial edits and commits numbers (or null). */
export function NumInput({ value, onChange, step, min, max, allowNull, className, sz = "md", placeholder, suffix }: {
  value: number | null | undefined; onChange: (v: number | null) => void; step?: number; min?: number; max?: number; allowNull?: boolean;
  className?: string; sz?: "sm" | "md"; placeholder?: string; suffix?: string;
}) {
  const [text, setText] = React.useState(value === null || value === undefined ? "" : String(value));
  React.useEffect(() => {
    const cur = text === "" ? null : Number(text);
    if (cur !== value) setText(value === null || value === undefined ? "" : String(value));
  }, [value]); // eslint-disable-line
  return (
    <div className={cn("relative", className)}>
      <input
        className={cn("field num pr-1", sz === "sm" && "field-sm", suffix && "pr-6")}
        value={text}
        placeholder={placeholder}
        inputMode="decimal"
        onChange={(e) => {
          const t = e.target.value;
          setText(t);
          if (t === "" || t === "-") {
            if (allowNull && t === "") onChange(null);
            return;
          }
          const n = Number(t);
          if (Number.isFinite(n)) onChange(n);
        }}
        onBlur={() => {
          let n = text === "" ? null : Number(text);
          if (n !== null && Number.isFinite(n)) {
            if (min !== undefined) n = Math.max(min, n);
            if (max !== undefined) n = Math.min(max, n);
            onChange(n);
            setText(String(n));
          } else if (!allowNull) setText(value === null || value === undefined ? "" : String(value));
        }}
        onKeyDown={(e) => {
          if (e.key === "ArrowUp" || e.key === "ArrowDown") {
            e.preventDefault();
            const s = step ?? 1;
            const n = (Number(text) || 0) + (e.key === "ArrowUp" ? s : -s);
            const r = Math.round(n * 1e6) / 1e6;
            setText(String(r));
            onChange(r);
          }
        }}
      />
      {suffix && <span className="absolute right-1.5 top-1/2 -translate-y-1/2 text-2xs text-faint pointer-events-none">{suffix}</span>}
    </div>
  );
}

export function Select<T extends string>({ value, onChange, options, className, sz = "md", disabled }: {
  value: T; onChange: (v: T) => void; options: (T | { value: T; label: string })[]; className?: string; sz?: "sm" | "md"; disabled?: boolean;
}) {
  return (
    <select disabled={disabled} className={cn("field appearance-none pr-5 bg-no-repeat cursor-pointer", sz === "sm" && "field-sm", className)}
      style={{ backgroundImage: "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='5'%3E%3Cpath d='M0 0l4 5 4-5z' fill='%237a8496'/%3E%3C/svg%3E\")", backgroundPosition: "right 6px center" }}
      value={value} onChange={(e) => onChange(e.target.value as T)}>
      {options.map((o) => {
        const v = typeof o === "string" ? o : o.value;
        const l = typeof o === "string" ? o : o.label;
        return <option key={v} value={v} className="bg-bg-2">{l}</option>;
      })}
    </select>
  );
}

export function Switch({ checked, onChange, className }: { checked: boolean; onChange: (v: boolean) => void; className?: string }) {
  return (
    <SwitchP.Root checked={checked} onCheckedChange={onChange} className={cn("relative h-3.5 w-6 shrink-0 rounded-full bg-bg-4 data-[state=checked]:bg-accent/80 transition-colors border border-line-2", className)}>
      <SwitchP.Thumb className="block h-2.5 w-2.5 rounded-full bg-fg transition-transform translate-x-[1px] data-[state=checked]:translate-x-[11px]" />
    </SwitchP.Root>
  );
}

export function Segmented<T extends string>({ value, onChange, options, className }: { value: T; onChange: (v: T) => void; options: { value: T; label: React.ReactNode; title?: string }[]; className?: string }) {
  return (
    <ToggleGroupP.Root type="single" value={value} onValueChange={(v) => v && onChange(v as T)} className={cn("inline-flex h-6 rounded-sm border border-line-2 bg-bg-2 p-px", className)}>
      {options.map((o) => (
        <ToggleGroupP.Item key={o.value} value={o.value} title={o.title}
          className="px-2 text-2xs font-medium text-muted rounded-[2px] data-[state=on]:bg-bg-4 data-[state=on]:text-fg hover:text-fg transition-colors">
          {o.label}
        </ToggleGroupP.Item>
      ))}
    </ToggleGroupP.Root>
  );
}

export const Tabs = TabsP.Root;
export function TabsList({ className, ...p }: TabsP.TabsListProps) {
  return <TabsP.List className={cn("flex items-end gap-0 border-b border-line px-2 h-8 shrink-0", className)} {...p} />;
}
export function TabsTrigger({ className, ...p }: TabsP.TabsTriggerProps) {
  return <TabsP.Trigger className={cn("h-8 px-3 text-xs text-muted border-b-2 border-transparent -mb-px hover:text-fg data-[state=active]:text-fg data-[state=active]:border-accent transition-colors flex items-center gap-1.5", className)} {...p} />;
}
export const TabsContent = ({ className, ...p }: TabsP.TabsContentProps) => <TabsP.Content className={cn("outline-none", className)} {...p} />;

export function Tip({ content, children, side = "top" }: { content: React.ReactNode; children: React.ReactNode; side?: "top" | "bottom" | "left" | "right" }) {
  if (!content) return <>{children}</>;
  return (
    <TooltipP.Root delayDuration={250}>
      <TooltipP.Trigger asChild>{children}</TooltipP.Trigger>
      <TooltipP.Portal>
        <TooltipP.Content side={side} sideOffset={4} className="z-[80] max-w-xs rounded-sm border border-line-2 bg-bg-3 px-2 py-1 text-xs text-fg shadow-xl">{content}</TooltipP.Content>
      </TooltipP.Portal>
    </TooltipP.Root>
  );
}
export const TooltipProvider = TooltipP.Provider;

export function Dialog({ open, onOpenChange, title, children, width = 560, footer }: { open: boolean; onOpenChange: (o: boolean) => void; title: React.ReactNode; children: React.ReactNode; width?: number; footer?: React.ReactNode }) {
  return (
    <DialogP.Root open={open} onOpenChange={onOpenChange}>
      <DialogP.Portal>
        <DialogP.Overlay className="fixed inset-0 z-50 bg-black/60" />
        <DialogP.Content style={{ width }} className="fixed left-1/2 top-[12vh] z-50 -translate-x-1/2 max-h-[76vh] flex flex-col rounded-md border border-line-2 bg-bg-1 shadow-2xl outline-none">
          <div className="flex items-center justify-between h-9 px-3 border-b border-line">
            <DialogP.Title className="text-sm font-medium">{title}</DialogP.Title>
            <DialogP.Close asChild><Button variant="ghost" size="icon"><X /></Button></DialogP.Close>
          </div>
          <div className="overflow-auto p-3">{children}</div>
          {footer && <div className="flex justify-end gap-2 px-3 py-2 border-t border-line">{footer}</div>}
        </DialogP.Content>
      </DialogP.Portal>
    </DialogP.Root>
  );
}

export const Popover = PopoverP.Root;
export const PopoverTrigger = PopoverP.Trigger;
export function PopoverContent({ className, ...p }: PopoverP.PopoverContentProps) {
  return (
    <PopoverP.Portal>
      <PopoverP.Content sideOffset={4} className={cn("z-[70] rounded-sm border border-line-2 bg-bg-2 p-1 shadow-2xl outline-none", className)} {...p} />
    </PopoverP.Portal>
  );
}

export const Menu = DropdownP.Root;
export const MenuTrigger = DropdownP.Trigger;
export function MenuContent({ className, ...p }: DropdownP.DropdownMenuContentProps) {
  return (
    <DropdownP.Portal>
      <DropdownP.Content sideOffset={4} className={cn("z-[70] min-w-40 rounded-sm border border-line-2 bg-bg-2 p-1 shadow-2xl", className)} {...p} />
    </DropdownP.Portal>
  );
}
export function MenuItem({ className, ...p }: DropdownP.DropdownMenuItemProps) {
  return <DropdownP.Item className={cn("flex items-center gap-2 rounded-[2px] px-2 py-1 text-xs text-fg-2 outline-none cursor-pointer data-[highlighted]:bg-bg-4 data-[highlighted]:text-fg [&_svg]:size-3.5", className)} {...p} />;
}
export const MenuLabel = ({ children }: { children: React.ReactNode }) => <DropdownP.Label className="px-2 pt-1.5 pb-1 label">{children}</DropdownP.Label>;
export const MenuSep = () => <DropdownP.Separator className="my-1 h-px bg-line" />;

export const Kbd = ({ children }: { children: React.ReactNode }) => (
  <kbd className="inline-flex h-4 min-w-4 items-center justify-center rounded-[2px] border border-line-2 bg-bg-3 px-1 font-mono text-[10px] text-muted">{children}</kbd>
);

export function Field({ label, children, className, hint }: { label: React.ReactNode; children: React.ReactNode; className?: string; hint?: React.ReactNode }) {
  return (
    <label className={cn("flex flex-col gap-1 min-w-0", className)}>
      <span className="label flex items-center gap-1">{label}{hint && <Tip content={hint}><span className="text-faint cursor-help normal-case">ⓘ</span></Tip>}</span>
      {children}
    </label>
  );
}
