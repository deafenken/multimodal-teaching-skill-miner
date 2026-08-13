"use client";

import {memo} from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

import {cn} from "@/lib/cn";
import {openSourcePreview, safeMarkdownUrl, SOURCE_PREVIEW_WINDOW} from "@/lib/markdown-security";

export const MarkdownMessage = memo(function MarkdownMessage({children, className}: {children: string; className?: string}) {
  return (
    <div className={cn("teachlab-markdown min-w-0 text-[15px] leading-7 text-[var(--app-text-soft)]", className)}>
      <ReactMarkdown
        skipHtml
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        urlTransform={safeMarkdownUrl}
        components={{
          a: ({href, children: linkChildren}) => href ? (
            <a
              href={href}
              target={href.startsWith("http") ? SOURCE_PREVIEW_WINDOW : undefined}
              onClick={href.startsWith("http") ? (event) => {
                event.preventDefault();
                openSourcePreview(href);
              } : undefined}
            >
              {linkChildren}
            </a>
          ) : <span>{linkChildren}</span>,
          img: ({alt}) => <span className="text-[var(--app-muted)]">[图片{alt ? `：${alt}` : ""}]</span>,
          input: ({checked, ...props}) => <input {...props} checked={checked} readOnly aria-label={checked ? "已完成" : "未完成"} />,
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
});
