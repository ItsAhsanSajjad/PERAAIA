import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { ErrorBoundary } from "./components/common/ErrorBoundary";

const inter = Inter({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800"],
  display: "swap",
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "Ask PERA Assistant",
  description: "Punjab Enforcement & Regulatory Authority — AI-powered document assistant for rules, regulations & governance.",
  icons: {
    icon: "/favicon.png",
    apple: "/pera_logo.png",
  },
};

// Viewport meta — explicitly set width=device-width so mobile browsers
// render at the phone's actual CSS width instead of treating the page
// as a desktop-width (≈980 px) layout and zooming out. Without this,
// the header appears microscopic on phones.
export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f5f3ef" },
    { media: "(prefers-color-scheme: dark)",  color: "#0b1020" },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="antialiased">
        <ErrorBoundary>
          {children}
        </ErrorBoundary>
      </body>
    </html>
  );
}
