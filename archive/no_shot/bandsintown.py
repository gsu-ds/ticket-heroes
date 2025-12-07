# %% [markdown]
# # Bandsintown Artist + Event Scraper (Full Corrected Script)
#
# **Key Correction:** Playwright `await` calls are now enclosed in an asynchronous
# `main()` function to ensure the entire pipeline runs successfully in a single block.
#
# Pipeline:
# 1. Setup dirs & constants
# 2. Define all helper functions (GraphQL, Playwright, Scrape, Flatten)
# 3. Define the main asynchronous execution block (`main()`)
# 4. Execute `main()`
#

# %%
import asyncio
import json
import string
import time
from pathlib import Path
from typing import List, Dict, Tuple, Set

import pandas as pd
import requests
from playwright.async_api import async_playwright
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from datetime import datetime as dt # Added for clarity

console = Console()

# -------------------------------------------------------------------
# 1. Directories & Constants
# -------------------------------------------------------------------
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"

for d in [DATA_DIR, RAW_DIR, EXTERNAL_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Where we'll save slug inventories and panels
SLUGS_PATH = EXTERNAL_DIR / "bandsintown_artist_slugs.parquet"
ARTIST_PANEL_PATH = PROCESSED_DIR / "bandsintown_artist_panel.parquet"
EVENT_PANEL_PATH = PROCESSED_DIR / "bandsintown_event_panel.parquet"

# Bandsintown endpoints / constants
BANDSINTOWN_BASE = "https://www.bandsintown.com"
GRAPHQL_URL = "https://graphql.bandsintown.com/"
BANDSINTOWN_SEARCH_URL = f"{BANDSINTOWN_BASE}/api/search"

# Simple UA header for HTTP calls
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

CURATED_URLS = [
    f"{BANDSINTOWN_BASE}/",
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

# -------------------------------------------------------------------
# 2. Async Playwright/Scrape Helpers
# -------------------------------------------------------------------

async def get_bandsintown_token_async() -> str:
    """
    Async: Scrapes the Bandsintown JWT token from GraphQL/other requests.
    """
    console.print(Panel("[cyan]Capturing Bandsintown token (async)...[/cyan]"))

    token_box = {"value": None}

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )

        page = await browser.new_page()

        async def handle_request(request):
            auth = request.headers.get("authorization")
            if auth and "Bearer" in auth and token_box["value"] is None:
                token_box["value"] = auth.split("Bearer ")[-1]

        page.on("request", handle_request)

        await page.goto(BANDSINTOWN_BASE, wait_until="networkidle", timeout=60000)
        await asyncio.sleep(3)

        await browser.close()

    token = token_box["value"]
    if not token:
        raise RuntimeError("Failed to extract Bandsintown token via Playwright.")

    console.print("[green]Token acquired via async Playwright.[/green]")
    return token

async def extract_artist_slugs_from_page_async(page) -> Set[str]:
    """
    Extract artist/event slugs from a page using client-side JS.
    """
    hrefs: List[str] = await page.evaluate(
        """
        () => {
            const anchors = Array.from(document.querySelectorAll('a'));
            return anchors
                .map(a => a.getAttribute('href'))
                .filter(href => href && (href.startsWith('/a/') || href.startsWith('/e/')));
        }
        """
    )

    slugs: Set[str] = set()
    for href in hrefs:
        # Remove leading slash and split
        parts = href.strip("/").split("/")
        candidate = parts[-1]
        # Basic sanity — must contain a dash, avoids pure IDs
        if candidate and "-" in candidate:
            slugs.add(candidate.lower())

    return slugs

async def collect_slugs_from_urls_async(urls: List[str]) -> Set[str]:
    """
    Async: visit each curated URL and collect artist/event slugs.
    """
    console.print(
        Panel(
            f"[cyan]Collecting artist slugs async from {len(urls)} curated pages[/cyan]",
            border_style="cyan",
        )
    )

    all_slugs: Set[str] = set()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )
        page = await browser.new_page()

        for url in urls:
            console.print(f"[blue]Visiting:[/blue] {url}")
            try:
                await page.goto(url, wait_until="networkidle", timeout=60000)
            except Exception as e:
                console.print(f"[red]Error visiting {url}: {e}[/red]")
                continue

            await asyncio.sleep(2) # Give scripts a moment to hydrate

            slugs = await extract_artist_slugs_from_page_async(page)
            console.print(f"[green]Found {len(slugs)} slugs on this page[/green]")

            all_slugs.update(slugs)

        await browser.close()

    console.print(
        Panel(
            f"[bold green]Total curated slugs: {len(all_slugs):,}[/bold green]",
            border_style="green",
        )
    )
    return all_slugs


def search_artists_by_prefix(prefix: str) -> Set[str]:
    """
    Use Bandsintown's search endpoint to discover artist slugs by prefix.
    """
    params = {"query": prefix}
    try:
        resp = requests.get(
            BANDSINTOWN_SEARCH_URL,
            params=params,
            headers=HEADERS,
            timeout=10,
        )
    except Exception as e:
        console.print(f"[red]Request error for prefix '{prefix}': {e}[/red]")
        return set()

    if resp.status_code != 200:
        console.print(
            f"[yellow]Non-200 ({resp.status_code}) for prefix '{prefix}' – skipping[/yellow]"
        )
        return set()

    try:
        data = resp.json()
    except Exception:
        console.print(f"[yellow]Failed to parse JSON for prefix '{prefix}'[/yellow]")
        return set()

    slugs: Set[str] = set()
    artists = data.get("artists", []) or []
    for a in artists:
        slug = a.get("slug")
        if slug:
            slugs.add(slug.lower())

    return slugs


def expand_slugs_via_search(prefixes: List[str]) -> Set[str]:
    """
    Iterate over prefixes and aggregate slugs from the search endpoint.
    """
    console.print(
        Panel(
            f"[cyan]Expanding artist slugs via search prefixes ({len(prefixes)} prefixes)[/cyan]",
            border_style="cyan",
        )
    )
    all_slugs: Set[str] = set()

    for p in prefixes:
        slugs = search_artists_by_prefix(p)
        console.print(f"[blue]Prefix '{p}' → {len(slugs)} slugs[/blue]")
        all_slugs.update(slugs)
        time.sleep(0.2)  # throttle a bit

    console.print(
        Panel(
            f"[bold green]Total slugs from search expansion: {len(all_slugs):,}[/bold green]",
            border_style="green",
        )
    )
    return all_slugs

# -------------------------------------------------------------------
# 3. GraphQL Helpers & Scraper
# -------------------------------------------------------------------

ARTIST_QUERY = """
query GetArtist($slug: String!) {
  artist(slug: $slug) {
    id
    name
    slug
    imageUrl
    facebookUrl
    instagramUrl
    twitterUrl
    websiteUrl
    trackerCount
    upcomingEventCount
    pastEventCount
  }
}
"""

EVENTS_QUERY = """
query GetArtistEvents($slug: String!) {
  artist(slug: $slug) {
    id
    name
    slug
    events(sort: {field: datetime, order: ASC}) {
      id
      datetime
      title
      url
      festival
      lineup {
        name
      }
      venue {
        id
        name
        city
        region
        country
        latitude
        longitude
      }
    }
  }
}
"""

def graphql_query(token: str, query: str, variables: Dict) -> Dict:
    """
    Minimal GraphQL client for Bandsintown.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"query": query, "variables": variables}

    r = requests.post(GRAPHQL_URL, headers=headers, json=payload, timeout=20)
    if r.status_code != 200:
        # Check if the token has expired
        if r.status_code == 401:
            raise RuntimeError("GraphQL 401: Token unauthorized/expired.")
        raise RuntimeError(f"GraphQL {r.status_code}: {r.text}")

    data = r.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data

def scrape_artist_with_token(
    token: str,
    slug: str,
    out_dir: Path = RAW_DIR / "bandsintown",
) -> Tuple[Dict, Dict]:
    """
    Fetch artist metadata + events via GraphQL and save raw JSON snapshots.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[blue]Scraping artist '{slug}'[/blue]")

    try:
        artist_data = graphql_query(token, ARTIST_QUERY, {"slug": slug})
        events_data = graphql_query(token, EVENTS_QUERY, {"slug": slug})
    except RuntimeError as e:
        console.print(f"[red]Error scraping '{slug}': {e}[/red]")
        return {}, {} # Return empty dicts on failure

    ts = int(time.time())
    with open(out_dir / f"{slug}_artist_{ts}.json", "w") as f:
        json.dump(artist_data, f, indent=2)
    with open(out_dir / f"{slug}_events_{ts}.json", "w") as f:
        json.dump(events_data, f, indent=2)

    return artist_data, events_data


def batch_scrape_artists(
    slugs: List[str],
    token: str,
    max_artists: int = 2000,
) -> Dict[str, Dict]:
    """
    Scrape up to max_artists artists with a single token.
    """
    console.print(
        Panel(
            f"[cyan]Batch scraping up to {max_artists:,} Bandsintown artists[/cyan]",
            border_style="cyan",
        )
    )

    results: Dict[str, Dict] = {}

    for i, slug in enumerate(slugs[:max_artists], start=1):
        # Gracefully handle token expiration during a long run
        try:
            artist_data, events_data = scrape_artist_with_token(token, slug)
            if artist_data and events_data:
                results[slug] = {"artist": artist_data, "events": events_data}
        except RuntimeError as e:
            if "Token unauthorized/expired" in str(e):
                 console.print(f"[bold red]TOKEN EXPIRED. Stopping batch scrape at {i-1} artists.[/bold red]")
                 break # Stop if token is dead
            else:
                raise e # Re-raise unexpected errors

        if i % 50 == 0:
            console.print(f"[green]Scraped {i:,} artists so far[/green]")
        time.sleep(0.1)  # gentle throttle

    console.print(
        Panel(
            f"[bold green]Done. Scraped {len(results):,} artists.[/bold green]",
            border_style="green",
        )
    )
    return results

# -------------------------------------------------------------------
# 4. Flattening Helpers
# -------------------------------------------------------------------

def flatten_artist_record(slug: str, artist_json: Dict) -> Dict:
    """
    Flatten a single artist GraphQL JSON blob into a row dict.
    """
    artist = (artist_json or {}).get("data", {}).get("artist", {}) or {}

    return {
        "artist_slug": slug,
        "artist_id": artist.get("id"),
        "artist_name": artist.get("name"),
        "image_url": artist.get("imageUrl"),
        "facebook_url": artist.get("facebookUrl"),
        "instagram_url": artist.get("instagramUrl"),
        "twitter_url": artist.get("twitterUrl"),
        "website_url": artist.get("websiteUrl"),
        "tracker_count": artist.get("trackerCount"),
        "upcoming_event_count": artist.get("upcomingEventCount"),
        "past_event_count": artist.get("pastEventCount"),
    }

def flatten_event_records(slug: str, events_json: Dict) -> List[Dict]:
    """
    Flatten GraphQL events JSON for a single artist into row dicts.
    """
    artist = (events_json or {}).get("data", {}).get("artist", {}) or {}
    events = artist.get("events", []) or []

    rows: List[Dict] = []
    for ev in events:
        venue = ev.get("venue", {}) or {}
        lineup = ev.get("lineup", []) or []

        rows.append(
            {
                "artist_slug": slug,
                "artist_name": artist.get("name"),
                "event_id": ev.get("id"),
                "event_title": ev.get("title"),
                "event_url": ev.get("url"),
                "event_datetime": ev.get("datetime"),
                "is_festival": bool(ev.get("festival")),
                "venue_id": venue.get("id"),
                "venue_name": venue.get("name"),
                "venue_city": venue.get("city"),
                "venue_region": venue.get("region"),
                "venue_country": venue.get("country"),
                "venue_latitude": venue.get("latitude"),
                "venue_longitude": venue.get("longitude"),
                "lineup_names": ", ".join(
                    [m.get("name") for m in lineup if m.get("name")]
                ),
            }
        )

    return rows

# -------------------------------------------------------------------
# 5. Main Execution Function
# -------------------------------------------------------------------

async def main(max_artists_to_scrape: int = 100):
    """
    The main asynchronous pipeline execution block.
    """
    console.print(Panel(f"[bold yellow]STARTING FULL BANDSINTOWN PIPELINE (Max Artists: {max_artists_to_scrape:,})[/bold yellow]"))

    # --- Phase 1: Slug Collection ---
    console.print("\n--- Phase 1: Collecting Slugs ---")
    slugs_curated = await collect_slugs_from_urls_async(CURATED_URLS)

    # Optional search expansion (using a small, representative sample for speed)
    two_letter_prefixes = [a + b for a in string.ascii_lowercase for b in string.ascii_lowercase]
    slugs_search = expand_slugs_via_search(two_letter_prefixes[:10]) # Limit to 10 prefixes for quick testing

    slugs_combined = set()
    slugs_combined.update(slugs_curated)
    slugs_combined.update(slugs_search)

    console.print(
        Panel(
            f"[bold cyan]Combined slug universe[/bold cyan]\n"
            f"Curated: {len(slugs_curated):,}\n"
            f"Search: {len(slugs_search):,}\n"
            f"Total unique: {len(slugs_combined):,}",
            border_style="cyan",
        )
    )

    slug_df = pd.DataFrame({"artist_slug": sorted(list(slugs_combined))})
    slug_df.to_parquet(SLUGS_PATH, index=False)
    console.print(f"[green]Saved slug inventory → {SLUGS_PATH} ({len(slug_df):,} rows)[/green]")
    slug_list = slug_df["artist_slug"].tolist()


    # --- Phase 2: Batch Scraping (requires token) ---
    console.print("\n--- Phase 2: Scraping Artist Data ---")
    if not slug_list:
        console.print("[bold red]No slugs collected. Aborting scrape.[/bold red]")
        return

    # 1. Get token
    token = await get_bandsintown_token_async()

    # 2. Scrape
    bandsintown_raw = batch_scrape_artists(slug_list, token, max_artists=max_artists_to_scrape)
    console.print(f"Total raw records scraped: {len(bandsintown_raw):,}")


    # --- Phase 3: Data Flattening & Saving ---
    console.print("\n--- Phase 3: Flattening Data ---")

    # Flatten Artist Panel
    artist_rows = []
    for slug, payload in bandsintown_raw.items():
        artist_rows.append(flatten_artist_record(slug, payload.get("artist")))

    artist_panel = pd.DataFrame(artist_rows)
    artist_panel.to_parquet(ARTIST_PANEL_PATH, index=False)
    console.print(f"[green]Saved artist panel → {ARTIST_PANEL_PATH} ({artist_panel.shape[0]:,} rows)[/green]")

    # Flatten Event Panel
    event_rows: List[Dict] = []
    for slug, payload in bandsintown_raw.items():
        event_rows.extend(flatten_event_records(slug, payload.get("events")))

    event_panel = pd.DataFrame(event_rows)

    if "event_datetime" in event_panel.columns:
        event_panel["event_datetime"] = pd.to_datetime(
            event_panel["event_datetime"], errors="coerce"
        )

    event_panel.to_parquet(EVENT_PANEL_PATH, index=False)
    console.print(f"[green]Saved event panel → {EVENT_PANEL_PATH} ({event_panel.shape[0]:,} rows)[/green]")

    console.print(Panel("[bold yellow]PIPELINE COMPLETE![/bold yellow]"))
    
    # Return the panels for immediate inspection
    return artist_panel, event_panel

# -------------------------------------------------------------------
# 6. Execution (Run this in a single cell)
# -------------------------------------------------------------------

# NOTE: Playwright requires an active event loop for the 'await' calls.
# If you run this block in a Jupyter/IPython environment, the event loop is usually active,
# so running `await main()` directly often works.
# For a standard Python script, you would use `asyncio.run(main())`.

# Since you are in a notebook environment, use this:
# Adjust `max_artists_to_scrape` to control the number of artists you scrape.
if __name__ == "__main__":
    # Scrape 100 artists initially to verify the pipeline works end-to-end
    artist_panel_final, event_panel_final = await main(max_artists_to_scrape=100)
    
    # Display final results
    console.print("\n[bold magenta]--- Final Artist Panel Snapshot ---[/bold magenta]")
    display(artist_panel_final.head())
    console.print("\n[bold magenta]--- Final Event Panel Snapshot ---[/bold magenta]")
    display(event_panel_final.head())