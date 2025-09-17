# Documentation Conventions

This project uses MkDocs Material with markdownlint.
Follow these conventions to keep docs consistent and warning‑free.

- Link styles
  - Prefer relative links for pages inside the docs (e.g., `../guides/face-enhancement.md`).
  - Use absolute GitHub links for repository files outside `docs/`
    (e.g., LICENSE) to avoid broken relative paths.
  - Use section anchors with `#heading-text` and ensure the target heading
    exists and is unique.
- Headings
  - Avoid duplicate headings at the same nesting level in a single page.
  - Keep titles concise; use sentence case (e.g., "Quick start").
- Code blocks
  - Surround fenced code blocks with a blank line above and below.
  - Use language hints (```bash,```python,```json) for syntax highlighting.
- Line length
  - Try to keep lines ≤ 100 chars. For long URLs, break the sentence across
    lines if needed.
- Images & assets
  - Place images under `docs/assets/` and refer relatively from the page.
  - Include alt text; keep images light.
- Admonitions
  - Use for important notes and warnings (e.g., `!!! note`, `!!! warning`).
- Cross‑refs to CLIs
  - The primary CLI is `restoria`; legacy shims (`gfpup`, `gfpgan-infer`)
    remain but should be called out as legacy only when relevant.
- Stability
  - If a feature is experimental, label it clearly and prefer soft‑failure
    language in examples.

---

If a link or build warning appears, prefer fixing the source page/link rather
than suppressing the warning.
