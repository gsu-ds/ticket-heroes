"""
Bandsintown Combined Slug Discovery Pipeline
--------------------------------------------
1. Extract JWT token via __NEXT_DATA__
2. Collect curated artist slugs from BI category pages (Playwright async)
3. Collect additional slugs via prefix search (semi-undocumented API)
4. Save final unified slug inventory to parquet

Stops BEFORE artist/event scraping (your request).
"""

import asyncio
import json
import string
import time
from pathlib import Path
from typing import List, Dict, Set

import pandas as pd
import requests
from playwright.async_api import async_playwright
from rich.console import Console
from rich.panel import Panel

console = Console()

# ---------------------------------------------------------
# Paths & Directories
# ---------------------------------------------------------
DATA_DIR = Path("data")
EXTERNAL_DIR = DATA_DIR / "external"
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

SLUGS_PATH = EXTERNAL_DIR / "bandsintown_artist_slugs.parquet"

# ---------------------------------------------------------
# Bandsintown Constants
# ---------------------------------------------------------
BANDSINTOWN_BASE = "https://www.bandsintown.com"
SEARCH_URL = f"{BANDSINTOWN_BASE}/api/search"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

CURATED_URLS = [
    f"{BANDSINTOWN_BASE}/c/top-artists",
    f"{BANDSINTOWN_BASE}/c/trending",
    f"{BANDSINTOWN_BASE}/c/new-music",
    f"{BANDSINTOWN_BASE}/c/top-sellers",
    f"{BANDSINTOWN_BASE}/c/festivals",
    f"{BANDSINTOWN_BASE}/c/hip-hop",
    f"{BANDSINTOWN_BASE}/c/pop",
    f"{BANDSINTOWN_BASE}/c/country",
    f"{BANDSINTOWN_BASE}/c/rock",
    f"{BANDSINTOWN_BASE}/c/edm",
]

# ======================================================================
# 1. TOKEN EXTRACTION (WORKING VERSION)
# ======================================================================
async def get_bandsintown_token_async() -> str:
    """
    Reliable stealth token extractor:
    - Loads Bandsintown Top Artists page
    - Uses anti-detection settings
    - Captures Authorization header from GraphQL calls
    """

    console.print(Panel("[cyan]Extracting Bandsintown token using STEALTH browser...[/cyan]"))

    token_box = {"value": None}

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=False,                     # <-- NOT HEADLESS (important!)
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ]
        )

        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
            locale="en-US",
            timezone_id="America/New_York",
        )

        page = await context.new_page()

        # Listen for XHRs
        async def on_request(request):
            auth = request.headers.get("authorization")
            if auth and "Bearer" in auth:
                token_box["value"] = auth.split("Bearer ")[-1]

        page.on("request", on_request)

        # Must use a page that triggers GraphQL fetches
        target_url = "https://www.bandsintown.com/c/top-artists"
        await page.goto(target_url, wait_until="domcontentloaded")

        # Wait up to 10 seconds for token to appear
        for _ in range(20):
            if token_box["value"]:
                break
            await asyncio.sleep(0.5)

        await browser.close()

    if not token_box["value"]:
        raise RuntimeError("🔥 Stealth mode failed to detect a Bandsintown token.")

    console.print("[bold green]SUCCESS — Token captured![/bold green]")

    return token_box["value"]


# ======================================================================
# 2. ASYNC SLUG EXTRACTION FROM CURATED PAGES
# ======================================================================
async def collect_slugs_from_curated_pages_async(urls: List[str]) -> Set[str]:
    console.print(
        Panel(
            f"[cyan]Collecting curated slugs from {len(urls)} pages[/cyan]",
            border_style="cyan",
        )
    )

    slugs = set()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox",
                  "--disable-dev-shm-usage", "--disable-gpu"]
        )
        page = await browser.new_page()

        for url in urls:
            console.print(f"[blue]Visiting:[/blue] {url}")

            await page.goto(url, wait_until="networkidle")

            found = await page.evaluate("""
                () => {
                    const links = Array.from(
                        document.querySelectorAll('a[href^="/a/"], a[href^="/artists/"], a[href^="/e/"]')
                    );
                    return links.map(a => a.getAttribute("href"));
                }
            """)

            cleaned = set()

            for href in found:
                if not href:
                    continue
                parts = href.strip("/").split("/")
                candidate = parts[-1].lower()
                if "-" in candidate:
                    cleaned.add(candidate)

            console.print(f"[green]Found {len(cleaned)} slugs on page[/green]")
            slugs.update(cleaned)

        await browser.close()

    console.print(f"[bold green]Total curated slugs: {len(slugs)}[/bold green]")
    return slugs

# ======================================================================
# 3. SEARCH-BASED SLUG EXPANSION
# ======================================================================
def search_artists_by_prefix(prefix: str) -> Set[str]:
    params = {"query": prefix}
    try:
        r = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=10)
    except:
        return set()

    if r.status_code != 200:
        return set()

    try:
        data = r.json()
    except:
        return set()

    slugs = set()
    for artist in data.get("artists", []):
        if "slug" in artist:
            slugs.add(artist["slug"].lower())

    return slugs


def expand_slugs_via_search(prefixes: List[str]) -> Set[str]:
    console.print(
        Panel(
            f"[cyan]Prefix search expansion ({len(prefixes)} prefixes)[/cyan]",
            border_style="cyan",
        )
    )

    all_slugs = set()

    for p in prefixes:
        found = search_artists_by_prefix(p)
        all_slugs.update(found)
        time.sleep(0.15)

    console.print(f"[green]Prefix-search slugs: {len(all_slugs)}[/green]")
    return all_slugs

# ======================================================================
# 4. MAIN PIPELINE (Stops before artist/event scraping)
# ======================================================================
async def run_slug_pipeline():
    console.print(Panel("[bold cyan]Running Bandsintown Slug Discovery Pipeline[/bold cyan]"))

    # Step 1 — extract token (ensures site is reachable)
    token = await get_bandsintown_token_async()
    console.print(f"[green]Token prefix:[/green] {token[:20]}...")

    # Step 2 — curated page slugs
    slugs_curated = await collect_slugs_from_curated_pages_async(CURATED_URLS)

    # Step 3 — search expansion (only first 50 prefixes for testing)
    two_letter_prefixes = [a + b for a in string.ascii_lowercase for b in string.ascii_lowercase]
    slugs_search = expand_slugs_via_search(two_letter_prefixes[:50])

    # Step 4 — combine
    combined = sorted(list(slugs_curated | slugs_search))
    slug_df = pd.DataFrame({"artist_slug": combined})

    # Step 5 — save
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    slug_df.to_parquet(SLUGS_PATH, index=False)

    console.print(
        Panel(
            f"[green]Saved slug universe → {SLUGS_PATH}\n"
            f"Total slugs: {len(slug_df):,}[/green]"
        )
    )

    return slug_df


# ======================================================================
# ENTRY POINT
# ======================================================================
if __name__ == "__main__":
    asyncio.run(run_slug_pipeline())
