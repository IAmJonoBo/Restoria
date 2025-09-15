import React from "react";

type Props = { before: string; after: string; alt?: string };

// Minimal accessible compare slider stub. Real implementation will handle
// drag/keyboard and letterboxing for different sizes.
export default function CompareSlider({ before, after, alt }: Props) {
  return (
    <div aria-label="Compare" role="group" style={{ display: "grid", gap: 8 }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        <figure>
          <img src={before} alt={alt ?? "Before"} style={{ maxWidth: "100%" }} />
          <figcaption>Before</figcaption>
        </figure>
        <figure>
          <img src={after} alt={alt ?? "After"} style={{ maxWidth: "100%" }} />
          <figcaption>After</figcaption>
        </figure>
      </div>
    </div>
  );
}

