# page_fetch.py
# Fast path for reading a single web page: one HTTP GET plus HTML-to-text, no browser and no
# LLM loop. Typically 1-2 seconds versus 60-90+ for a browser-use run. Pages that need
# JavaScript, a login, or that block bots should go through browse_web instead.
#
# Only public http(s) URLs are fetched. Loopback, private, link-local and other internal
# addresses are refused (including after redirects), so a page or a prompt can't point the
# agent at services on this machine or the local network.

import ipaddress
import re
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

MAX_BYTES = 3_000_000       # stop reading huge responses
MAX_TEXT_CHARS = 20_000     # keep the model's input bounded
MAX_REDIRECTS = 5
TIMEOUT_SECONDS = 15
MIN_USEFUL_CHARS = 200      # less text than this usually means a JavaScript-rendered page
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")

_SKIP_TAGS = {"script", "style", "noscript", "svg", "template", "iframe", "head"}
_BLOCK_TAGS = {"p", "div", "section", "article", "header", "footer", "nav", "main", "aside",
               "li", "ul", "ol", "tr", "table", "br", "h1", "h2", "h3", "h4", "h5", "h6",
               "blockquote", "pre", "dd", "dt", "figcaption"}


class PageFetchError(Exception):
    pass


def check_public_url(url: str) -> None:
    """Raise PageFetchError unless url is http(s) and every address its host resolves to is public."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise PageFetchError(f"Only http and https URLs can be fetched (got '{parsed.scheme}').")
    host = parsed.hostname
    if not host:
        raise PageFetchError("The URL has no host.")
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as e:
        raise PageFetchError(f"Could not resolve {host}: {e}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise PageFetchError(f"Refusing to fetch {host}: it resolves to a non-public address ({ip}).")


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self.description = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            a = dict(attrs)
            if (a.get("name") or a.get("property") or "").lower() in ("description", "og:description"):
                self.description = self.description or (a.get("content") or "").strip()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in raw.split("\n"))
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def html_to_text(html: str) -> tuple[str, str, str]:
    """Return (title, meta description, visible text) for an HTML document."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.title.strip(), parser.description, parser.text()


async def fetch_page_text(url: str) -> dict:
    """Fetch a public page and return {'final_url', 'status', 'title', 'description', 'text',
    'truncated', 'sparse'}. Raises PageFetchError for refused or failed fetches."""
    current = url
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=False,
                                 headers={"User-Agent": USER_AGENT,
                                          "Accept": "text/html,application/xhtml+xml"}) as client:
        for _ in range(MAX_REDIRECTS + 1):
            check_public_url(current)
            try:
                async with client.stream("GET", current) as resp:
                    if resp.is_redirect and resp.headers.get("location"):
                        current = urljoin(current, resp.headers["location"])
                        continue
                    body = b""
                    async for chunk in resp.aiter_bytes():
                        body += chunk
                        if len(body) >= MAX_BYTES:
                            break
                    status = resp.status_code
                    ctype = resp.headers.get("content-type", "")
                    encoding = resp.encoding or "utf-8"
            except httpx.HTTPError as e:
                raise PageFetchError(f"Request to {current} failed: {e}")
            break
        else:
            raise PageFetchError(f"Too many redirects starting from {url}.")

    if status >= 400:
        raise PageFetchError(f"{current} returned HTTP {status} (the site may block simple fetches).")
    html = body.decode(encoding, "replace")
    if "html" not in ctype.lower() and not html.lstrip().startswith("<"):
        text, title, description = html, "", ""
    else:
        title, description, text = html_to_text(html)
    truncated = len(text) > MAX_TEXT_CHARS
    return {"final_url": current, "status": status, "title": title, "description": description,
            "text": text[:MAX_TEXT_CHARS], "truncated": truncated,
            "sparse": len(text) < MIN_USEFUL_CHARS}
