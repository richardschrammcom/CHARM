"""Fast page fetch: refuses internal addresses and extracts visible text."""
import asyncio

import pytest

import page_fetch
from page_fetch import PageFetchError, check_public_url, html_to_text


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:7060/",
    "https://localhost:7004/",
    "http://10.0.0.5/",
    "http://192.168.1.1/",
    "http://169.254.169.254/latest/meta-data/",
    "file:///etc/passwd",
    "ftp://example.com/",
])
def test_refuses_non_public_urls(url):
    with pytest.raises(PageFetchError):
        check_public_url(url)


def test_html_to_text_skips_scripts_and_reads_meta():
    html = ("<html><head><title>QCC</title><meta name='description' content='A conference'>"
            "<script>var x = 'hidden';</script><style>p{}</style></head>"
            "<body><nav>Menu</nav><p>Nov 13&ndash;15</p><p>Cincinnati</p></body></html>")
    title, desc, text = html_to_text(html)
    assert title == "QCC"
    assert desc == "A conference"
    assert "hidden" not in text and "p{}" not in text
    assert "Nov 13–15" in text and "Cincinnati" in text


def test_redirect_to_internal_address_is_refused(monkeypatch):
    calls = []

    def fake_check(url):
        calls.append(url)
        if "127.0.0.1" in url:
            raise PageFetchError("internal")

    class FakeResp:
        is_redirect = True
        headers = {"location": "http://127.0.0.1:7060/"}
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    monkeypatch.setattr(page_fetch, "check_public_url", fake_check)
    monkeypatch.setattr(page_fetch.httpx.AsyncClient, "stream", lambda self, m, u: FakeResp())
    with pytest.raises(PageFetchError):
        asyncio.run(page_fetch.fetch_page_text("https://example.com/"))
    assert calls == ["https://example.com/", "http://127.0.0.1:7060/"]
