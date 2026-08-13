import type {Metadata} from "next";
import type {ReactNode} from "react";

import {Providers} from "@/app/providers";
import {ServiceWorkerLifecycle} from "@/components/service-worker-lifecycle";
import "@/app/globals.css";

export const metadata: Metadata = {
  title: "TeachLab Agent Console",
  description: "Adaptive teaching agent workspace with a docked inspector"
};

export default function RootLayout({children}: {children: ReactNode}) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var value=JSON.parse(localStorage.getItem("teachlab.console.preferences")||"{}");document.documentElement.dataset.consoleTheme=value.theme==="Light"?"light":"dark";document.documentElement.dataset.consoleDensity=value.density==="Compact"?"compact":"default";document.documentElement.dataset.consoleReduceMotion=String(value.reduceMotion===true)}catch(_){document.documentElement.dataset.consoleTheme="dark";document.documentElement.dataset.consoleDensity="default";document.documentElement.dataset.consoleReduceMotion="false"}})();`,
          }}
        />
      </head>
      <body>
        <ServiceWorkerLifecycle />
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
