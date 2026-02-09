import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "PERA AI Assistant",
  description: "Punjab Enforcement & Regulatory Authority - AI Assistant",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.className}>
      <head>
        <meta name="theme-color" content="#0a0f1a" />
      </head>
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
