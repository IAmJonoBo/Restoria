import "../styles/theme.css";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <main style={{ maxWidth: 1100, margin: "0 auto", padding: 16 }}>{children}</main>
      </body>
    </html>
  );
}

