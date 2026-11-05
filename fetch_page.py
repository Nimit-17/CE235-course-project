"""Fetch visible page text. Prefer Playwright; fall back to HTTP."""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
PROFILE = ROOT / ".pw-profile"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

BLOCK_PHRASES = (
    "access denied",
    "request blocked",
    "you have been blocked",
    "just a moment",
    "performing security verification",
    "attention required",
    "enable javascript and cookies",
    "cf-browser-verification",
)


def looks_blocked(text: str) -> bool:
    lowered = text.lower()
    if len(text.strip()) < 40:
        return True
    return any(phrase in lowered for phrase in BLOCK_PHRASES)


def clean_text(value: str) -> str:
    return " ".join((value or "").split())


def http_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            raw = response.read().decode("utf-8", "replace")
            final = response.geturl()
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", "replace")
        final = url
    raw = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    raw = re.sub(r"(?is)<[^>]+>", " ", raw)
    return clean_text(raw), final


def posting_gone_from_url(requested: str, final: str) -> bool:
    """Greenhouse soft-redirects dead jobs to ?error=true or the board root."""
    final_l = (final or "").lower()
    req = urlparse(requested)
    fin = urlparse(final or requested)
    qs = parse_qs(fin.query)
    if qs.get("error", [""])[0].lower() == "true":
        return True
    if "/jobs/" in req.path.lower() and "/jobs/" not in fin.path.lower():
        return True
    if "job-boards.greenhouse.io" in final_l and "/jobs/" not in fin.path.lower():
        return True
    return False


def _collect_dialog_text(page) -> str:
    parts = []
    selectors = [
        "[role='dialog']",
        "[role='alertdialog']",
        "[role='alert']",
        ".toast",
        ".Toast",
        "[class*='toast']",
        "[class*='Toast']",
        "[class*='banner']",
        "[class*='Banner']",
        "[class*='modal']",
        "[class*='Modal']",
        "[class*='flash']",
    ]
    for selector in selectors:
        try:
            nodes = page.locator(selector)
            count = min(nodes.count(), 6)
            for index in range(count):
                try:
                    text = clean_text(nodes.nth(index).inner_text(timeout=500))
                except Exception:
                    continue
                if not text or text in parts:
                    continue
                low = text.lower()
                if "results found" in low and "afghanistan" in low:
                    continue
                if low.count("+") > 20:
                    continue
                parts.append(text)
        except Exception:
            continue
    return clean_text(" ".join(parts))


def _read_body(page) -> str:
    try:
        return clean_text(page.inner_text("body"))
    except Exception:
        return ""


def playwright_fetch(url: str, timeout_ms: int = 45000) -> dict:
    from playwright.sync_api import sync_playwright

    PROFILE.mkdir(exist_ok=True)
    last_error = None
    attempts = [
        {"channel": "msedge", "headless": True},
        {"channel": "msedge", "headless": False},
        {"headless": True},
    ]
    with sync_playwright() as playwright:
        for options in attempts:
            user_dir = PROFILE / (
                "edge" if options.get("channel") == "msedge" else "chromium"
            )
            user_dir.mkdir(exist_ok=True)
            try:
                context = playwright.chromium.launch_persistent_context(
                    user_data_dir=str(user_dir),
                    viewport={"width": 1280, "height": 900},
                    user_agent=UA,
                    args=["--disable-blink-features=AutomationControlled"],
                    ignore_default_args=["--enable-automation"],
                    **options,
                )
            except Exception as exc:
                last_error = exc
                continue

            page = context.new_page()
            early = ""
            dialog = ""
            late = ""
            final_url = url
            try:
                page.add_init_script(
                    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
                )
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                page.wait_for_timeout(800)
                early = _read_body(page)
                dialog = _collect_dialog_text(page)
                page.wait_for_timeout(3500)
                late = _read_body(page)
                dialog2 = _collect_dialog_text(page)
                if dialog2:
                    dialog = clean_text(f"{dialog} {dialog2}")
                final_url = page.url
            finally:
                context.close()

            # Prefer a non-blocked late page; keep early/dialog even if late is a board.
            candidate = late or early
            if candidate and not looks_blocked(candidate):
                return {
                    "text": late or early,
                    "early_text": early,
                    "dialog_text": dialog,
                    "final_url": final_url,
                    "method": "playwright",
                }
            # Soft-closed Greenhouse often still has useful early/dialog text.
            if (early or dialog) and not looks_blocked(early or dialog):
                return {
                    "text": late or early or dialog,
                    "early_text": early,
                    "dialog_text": dialog,
                    "final_url": final_url,
                    "method": "playwright",
                }
            last_error = RuntimeError(
                f"blocked or empty via {options}: {(candidate or '')[:120]}"
            )
    if last_error:
        raise last_error
    return {
        "text": "",
        "early_text": "",
        "dialog_text": "",
        "final_url": url,
        "method": "playwright",
    }


def fetch_page(url: str) -> dict:
    """
    Returns ok/blocked/text/early_text/dialog_text/final_url/posting_gone/method/error.
    """
    payload = {
        "text": "",
        "early_text": "",
        "dialog_text": "",
        "final_url": url,
        "posting_gone": False,
        "method": None,
        "error": None,
    }
    try:
        fetched = playwright_fetch(url)
        payload.update(fetched)
        payload["method"] = "playwright"
    except Exception as exc:
        payload["error"] = f"playwright: {exc}"
        try:
            text, final = http_text(url)
            payload["text"] = text
            payload["early_text"] = text[:800]
            payload["final_url"] = final
            payload["method"] = "http"
            payload["error"] = f"{payload['error']}; fell back to http"
        except Exception as exc2:
            return {
                "ok": False,
                "blocked": True,
                "posting_gone": False,
                "text": "",
                "early_text": "",
                "dialog_text": "",
                "final_url": url,
                "method": None,
                "error": f"{payload['error']}; http: {exc2}",
            }

    text = payload.get("text") or ""
    early = payload.get("early_text") or ""
    dialog = payload.get("dialog_text") or ""
    final_url = payload.get("final_url") or url
    posting_gone = posting_gone_from_url(url, final_url)
    blocked = looks_blocked(text) and looks_blocked(early or dialog or text)
    ok = (not blocked and len(text) >= 40) or posting_gone or (
        len(early) >= 40 and not looks_blocked(early)
    )
    return {
        "ok": ok,
        "blocked": blocked and not posting_gone,
        "posting_gone": posting_gone,
        "text": text,
        "early_text": early,
        "dialog_text": dialog,
        "final_url": final_url,
        "method": payload.get("method"),
        "error": payload.get("error"),
    }


if __name__ == "__main__":
    import json
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else (
        "https://job-boards.greenhouse.io/enterpret/jobs/4592214005"
    )
    result = fetch_page(target)
    printable = {
        key: (value[:350] if key in {"text", "early_text", "dialog_text"} and isinstance(value, str) else value)
        for key, value in result.items()
    }
    print(json.dumps(printable, indent=2))
