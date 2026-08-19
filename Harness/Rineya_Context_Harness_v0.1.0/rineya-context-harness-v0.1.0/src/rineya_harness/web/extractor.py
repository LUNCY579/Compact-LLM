from __future__ import annotations

import re

from bs4 import BeautifulSoup

_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


class ReadableTextExtractor:
    """Conservative HTML-to-text extraction for small-model ingestion."""

    def extract(self, html: str) -> tuple[str | None, str]:
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else None
        for element in soup(
            [
                "script",
                "style",
                "noscript",
                "svg",
                "canvas",
                "iframe",
                "form",
                "nav",
                "footer",
                "aside",
            ]
        ):
            element.decompose()
        root = soup.find("main") or soup.find("article") or soup.body or soup
        lines: list[str] = []
        for raw in root.get_text("\n").splitlines():
            cleaned = _WHITESPACE.sub(" ", raw).strip()
            if cleaned:
                lines.append(cleaned)
        text = _MANY_NEWLINES.sub("\n\n", "\n".join(lines)).strip()
        return title, text
