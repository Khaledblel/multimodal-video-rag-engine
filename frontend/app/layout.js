import "./globals.css";

export const metadata = {
  title: "Video RAG",
  description: "Ask questions about your videos",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
