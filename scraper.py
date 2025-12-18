import argparse
import csv
import hashlib
import logging
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from tqdm.auto import tqdm
except Exception:

    def tqdm(iterable: Iterable, **_: Dict) -> Iterable:
        return iterable


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# Configuration
DATA_DIR = Path("data")
MASTER_CSV_PATH = DATA_DIR / "master_jobs.csv"
DELTA_CSV_PATH = DATA_DIR / "delta_jobs.csv"

KEYWORDS = [
    "ai",
    "artificial intelligence",
    "ml",
    "machine learning",
    "engineer",
    "developer",
]

EUROPE_REMOTE_HINTS = [
    "remote",
    "europe",
    "eu",
    "germany",
    "france",
    "uk",
    "united kingdom",
    "england",
    "spain",
    "italy",
    "netherlands",
    "belgium",
    "sweden",
    "norway",
    "finland",
    "denmark",
    "switzerland",
    "austria",
    "ireland",
    "portugal",
    "poland",
    "czech",
    "estonia",
    "lithuania",
    "latvia",
]

ATS_PLATFORMS = [
    ("jobs.lever.co", "lever"),
    ("jobs.ashbyhq.com", "ashby"),
    ("boards.greenhouse.io", "greenhouse"),
    ("jobs.smartrecruiters.com", "smartrecruiters"),
    ("wd1.myworkdayjobs.com", "workday"),
    ("jobs.bamboohr.com", "bamboohr"),
    ("jobs.jobvite.com", "jobvite"),
    ("careers.icims.com", "icims"),
    ("apply.jazz.co", "jazz"),
    ("careers.workable.com", "workable"),
]

ATS_DOMAINS = [domain for domain, _ in ATS_PLATFORMS]

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ATSJobScraper/1.0; +https://example.com)",
    "Accept-Language": "en-US,en;q=0.9",
}


# Persistence Layer


def generate_job_id(title: str, company: str, location: str, url: str) -> str:
    """Generate deterministic job ID from job attributes."""
    base = "||".join(
        [
            (title or "").strip().lower(),
            (company or "").strip().lower(),
            (location or "").strip().lower(),
            (url or "").strip().lower(),
        ]
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def load_master_csv(path: Path = MASTER_CSV_PATH) -> pd.DataFrame:
    """Load master CSV if exists, otherwise return empty DataFrame."""
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "job_id",
                "title",
                "company",
                "location",
                "description",
                "url",
                "source",
                "scraped_at",
            ]
        )
    return pd.read_csv(path, dtype=str)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV with proper quoting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL)


def compute_delta(master_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    """Return jobs from current_df that are not in master_df."""
    if master_df.empty:
        return current_df.copy()

    master_ids = set(master_df["job_id"].astype(str))
    mask = ~current_df["job_id"].astype(str).isin(master_ids)
    return current_df[mask].copy()


# HTTP Layer


def fetch_url(url: str) -> Optional[str]:
    """Fetch URL content with error handling."""
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        if response.status_code != 200:
            logger.warning("Non-200 status %s for URL %s", response.status_code, url)
            return None
        return response.text
    except requests.RequestException:
        logger.exception("Request failed for URL %s", url)
        return None


# ============================================================================
# Discovery Layer (Search API)
# ============================================================================


def _extract_urls_from_serpapi_response(
    data: dict, platform_domain: str
) -> List[tuple]:
    """Extract (url, title) tuples from SerpAPI JSON response."""
    urls = []
    if "organic_results" in data:
        for result in data["organic_results"]:
            url = result.get("link", "")
            title = result.get("title", "")
            if url and platform_domain in url:
                urls.append((url, title))
    return urls


def _extract_urls_from_bing_response(data: dict, platform_domain: str) -> List[tuple]:
    """Extract (url, title) tuples from Bing API JSON response."""
    urls = []
    if "webPages" in data and "value" in data["webPages"]:
        for result in data["webPages"]["value"]:
            url = result.get("url", "")
            title = result.get("name", "")
            if url and platform_domain in url:
                urls.append((url, title))
    return urls


def _query_search_api(
    search_query: str,
    platform_domain: str,
    serpapi_key: Optional[str],
    bing_api_key: Optional[str],
) -> tuple[List[tuple], Optional[str]]:
    """
    Query search API (SerpAPI or Bing) and return (url, title) pairs.
    Returns (urls_found, api_used).
    """
    urls_found = []
    api_used = None

    # Try SerpAPI first
    if serpapi_key:
        try:
            response = requests.get(
                "https://serpapi.com/search",
                params={
                    "q": search_query,
                    "api_key": serpapi_key,
                    "num": 50,
                    "engine": "google",
                },
                timeout=20,
            )
            if response.status_code == 200:
                data = response.json()
                api_used = "SerpAPI"
                urls_found = _extract_urls_from_serpapi_response(data, platform_domain)
                logger.info(
                    "SerpAPI returned %d results for %s",
                    len(data.get("organic_results", [])),
                    platform_domain,
                )
            elif response.status_code == 401:
                logger.warning("SerpAPI key invalid or expired")
            elif response.status_code == 429:
                logger.warning("SerpAPI quota exceeded")
            else:
                logger.debug(
                    "SerpAPI returned status %d for %s",
                    response.status_code,
                    platform_domain,
                )
        except requests.RequestException as e:
            logger.debug("SerpAPI request failed for %s: %s", platform_domain, e)
        except Exception as e:
            logger.debug("SerpAPI parsing failed for %s: %s", platform_domain, e)

    # Fallback to Bing API if SerpAPI failed or not configured
    if not urls_found and bing_api_key:
        try:
            response = requests.get(
                "https://api.bing.microsoft.com/v7.0/search",
                headers={"Ocp-Apim-Subscription-Key": bing_api_key},
                params={"q": search_query, "count": 50},
                timeout=20,
            )
            if response.status_code == 200:
                data = response.json()
                api_used = "Bing API"
                urls_found = _extract_urls_from_bing_response(data, platform_domain)
                logger.info(
                    "Bing API returned %d results for %s",
                    len(data.get("webPages", {}).get("value", [])),
                    platform_domain,
                )
            elif response.status_code == 401:
                logger.warning("Bing API key invalid")
            elif response.status_code == 429:
                logger.warning("Bing API quota exceeded")
            else:
                logger.debug(
                    "Bing API returned status %d for %s",
                    response.status_code,
                    platform_domain,
                )
        except requests.RequestException as e:
            logger.debug("Bing API request failed for %s: %s", platform_domain, e)
        except Exception as e:
            logger.debug("Bing API parsing failed for %s: %s", platform_domain, e)

    return urls_found, api_used


def discover_job_urls_via_search_api(
    query_keywords: List[str], location_keywords: List[str], max_results: int = 100
) -> List[Dict[str, str]]:
    """
    Discover job URLs via search API (SerpAPI or Bing).
    Returns list of job dicts with url, title, company, source (location/description empty).
    """
    logger.info("Searching for jobs via search API...")

    serpapi_key = os.getenv("SERPAPI_KEY")
    bing_api_key = os.getenv("BING_SEARCH_API_KEY")

    if not serpapi_key and not bing_api_key:
        logger.error(
            "No search API key found. Set SERPAPI_KEY or BING_SEARCH_API_KEY environment variable."
        )
        return []

    # Build search query parts (limit to respect API constraints)
    limited_keywords = query_keywords[:3]
    limited_locations = location_keywords[:4]
    keyword_part = " OR ".join([f'"{kw}"' for kw in limited_keywords])
    location_part = " OR ".join([f'"{loc}"' for loc in limited_locations])

    discovered_jobs = []
    seen_urls = set()

    for platform_domain, source_name in ATS_PLATFORMS:
        try:
            search_query = (
                f"site:{platform_domain} ({keyword_part}) AND ({location_part})"
            )
            logger.info("Searching %s with query: %s", platform_domain, search_query)

            urls_found, api_used = _query_search_api(
                search_query, platform_domain, serpapi_key, bing_api_key
            )

            found_count = 0
            for url, title in urls_found:
                if url in seen_urls:
                    continue
                if "/search" in url.lower() or "?q=" in url:
                    continue

                seen_urls.add(url)

                # Extract company name from URL
                company = "Unknown"
                if platform_domain == "jobs.lever.co":
                    match = re.search(r"jobs\.lever\.co/([^/]+)", url)
                    if match:
                        company = match.group(1)
                elif platform_domain == "jobs.ashbyhq.com":
                    match = re.search(r"jobs\.ashbyhq\.com/([^/]+)", url)
                    if match:
                        company = match.group(1)

                discovered_jobs.append(
                    {
                        "url": url,
                        "title": title or "Job Posting",
                        "company": company,
                        "location": "",
                        "description": "",
                        "source": source_name,
                    }
                )
                found_count += 1

            if found_count > 0:
                logger.info(
                    "Found %d job URLs from %s via %s",
                    found_count,
                    platform_domain,
                    api_used or "Unknown",
                )

            time.sleep(1)  # Rate limiting

        except Exception:
            logger.exception("Error searching %s", platform_domain)
            continue

    logger.info("Found %d total job URLs via search API", len(discovered_jobs))
    if len(discovered_jobs) == 0:
        logger.warning(
            "Zero job URLs discovered. Possible causes: API quota exceeded, invalid API key, or no matching jobs."
        )

    return discovered_jobs[:max_results]


# ============================================================================
# Enrichment Layer (Web Scraping)
# ============================================================================


def _parse_lever_job_page(soup: BeautifulSoup, url: str) -> Dict[str, str]:
    """Extract job data from Lever job page."""
    job_data = {}

    title_el = soup.find("h2", class_="posting-headline") or soup.find("h1")
    if title_el:
        job_data["title"] = title_el.get_text(strip=True)

    location_el = soup.find("div", class_="posting-categories") or soup.find(
        "span", class_="sort-by-location"
    )
    if location_el:
        job_data["location"] = location_el.get_text(strip=True)

    desc_el = soup.find("div", class_="section-wrapper") or soup.find(
        "div", class_="posting"
    )
    if desc_el:
        job_data["description"] = " ".join(
            desc_el.get_text(separator=" ", strip=True).split()[:500]
        )

    # Extract company from URL if needed
    match = re.search(r"jobs\.lever\.co/([^/]+)", url)
    if match:
        job_data["company"] = match.group(1)

    return job_data


def _parse_ashby_job_page(soup: BeautifulSoup, url: str) -> Dict[str, str]:
    """Extract job data from Ashby job page."""
    job_data = {}

    title_el = soup.find("h1") or soup.find("title")
    if title_el:
        job_data["title"] = title_el.get_text(strip=True)

    location_el = soup.find("div", attrs={"data-testid": "job-location"}) or soup.find(
        "span", class_="location"
    )
    if location_el:
        job_data["location"] = location_el.get_text(strip=True)

    desc_el = (
        soup.find("section", attrs={"data-testid": "job-description"})
        or soup.find("div", attrs={"data-testid": "job-description"})
        or soup.find("main")
    )
    if desc_el:
        job_data["description"] = " ".join(
            desc_el.get_text(separator=" ", strip=True).split()[:500]
        )

    # Extract company from URL if needed
    match = re.search(r"jobs\.ashbyhq\.com/([^/]+)", url)
    if match:
        job_data["company"] = match.group(1)

    return job_data


def _parse_generic_job_page(soup: BeautifulSoup) -> Dict[str, str]:
    """Extract job data from generic ATS job page (fallback parser)."""
    job_data = {}

    title_el = soup.find("h1") or soup.find("title")
    if title_el:
        job_data["title"] = title_el.get_text(strip=True)

    desc_el = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_="description")
    )
    if desc_el:
        job_data["description"] = " ".join(
            desc_el.get_text(separator=" ", strip=True).split()[:500]
        )

    return job_data


def enrich_job_from_url(job: Dict[str, str]) -> Dict[str, str]:
    """
    Enrich job dict by scraping the job URL for full details.
    Lever and Ashby use platform-specific parsers; other platforms use generic parser.
    """
    url = job.get("url", "")
    if not url or not url.startswith("http"):
        return job

    html = fetch_url(url)
    if not html:
        return job

    soup = BeautifulSoup(html, "html.parser")
    source = job.get("source", "")

    # Platform-specific parsing
    if source == "lever":
        enriched = _parse_lever_job_page(soup, url)
    elif source == "ashby":
        enriched = _parse_ashby_job_page(soup, url)
    else:
        # Generic parser for all other ATS platforms
        enriched = _parse_generic_job_page(soup)

    # Merge enriched data into job dict (only update non-empty fields)
    for key, value in enriched.items():
        if value:
            job[key] = value

    return job


# ============================================================================
# Filtering Layer
# ============================================================================


def passes_keyword_filter(title: str, description: str) -> bool:
    """Return True if job matches keyword filter."""
    combined = f"{title} {description}".lower()
    return any(keyword in combined for keyword in KEYWORDS)


def passes_location_filter(location: str, description: str) -> bool:
    """Return True if job matches location filter."""
    text = f"{location} {description}".lower()
    return any(hint in text for hint in EUROPE_REMOTE_HINTS)


# ============================================================================
# Normalization Layer
# ============================================================================


def normalize_jobs(raw_jobs: List[Dict[str, str]]) -> pd.DataFrame:
    """Convert raw job dicts to normalized DataFrame with job_id and scraped_at."""
    now_iso = datetime.now(UTC).isoformat()

    normalized = []
    for job in raw_jobs:
        job_id = generate_job_id(
            title=job.get("title", ""),
            company=job.get("company", ""),
            location=job.get("location", ""),
            url=job.get("url", ""),
        )
        normalized.append(
            {
                "job_id": job_id,
                "title": job.get("title", ""),
                "company": job.get("company", ""),
                "location": job.get("location", ""),
                "description": job.get("description", ""),
                "url": job.get("url", ""),
                "source": job.get("source", ""),
                "scraped_at": now_iso,
            }
        )

    if not normalized:
        return pd.DataFrame(
            columns=[
                "job_id",
                "title",
                "company",
                "location",
                "description",
                "url",
                "source",
                "scraped_at",
            ]
        )

    return pd.DataFrame(normalized)


# ============================================================================
# Main Pipeline
# ============================================================================


def _extract_query_parameters(query: str) -> tuple[List[str], List[str]]:
    """Extract keywords and location hints from user query."""
    query_keywords = KEYWORDS.copy()
    location_keywords = EUROPE_REMOTE_HINTS.copy()

    query_lower = query.lower()
    if (
        "ai" in query_lower
        or "artificial intelligence" in query_lower
        or "ml" in query_lower
    ):
        query_keywords = [
            "AI",
            "Artificial Intelligence",
            "ML",
            "Machine Learning",
            "Engineer",
            "Developer",
        ]
    if "engineer" in query_lower or "developer" in query_lower:
        query_keywords.extend(["Engineer", "Developer"])
    if (
        "europe" in query_lower
        or "eu" in query_lower
        or "uk" in query_lower
        or "remote" in query_lower
    ):
        location_keywords = ["Europe", "EU", "UK", "remote"] + EUROPE_REMOTE_HINTS

    return query_keywords, location_keywords


def main() -> None:
    """Main entry point: Search API → Enrich → Filter → Normalize → Delta → CSV."""
    parser = argparse.ArgumentParser(
        description="Weekly ATS Job Scraper using search API discovery."
    )
    parser.add_argument(
        "query",
        type=str,
        help='Free-form search query (e.g. "AI engineer roles in Europe / remote").',
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting ATS job scrape...")
    logger.info("Search query: %s", args.query)

    # Check API keys
    serpapi_key = os.getenv("SERPAPI_KEY")
    bing_api_key = os.getenv("BING_SEARCH_API_KEY")
    if not serpapi_key and not bing_api_key:
        logger.error(
            "No search API key found. Set SERPAPI_KEY or BING_SEARCH_API_KEY environment variable."
        )
        logger.error(
            "Get SerpAPI key: https://serpapi.com/ (recommended) or Bing API key: https://www.microsoft.com/en-us/bing/apis/bing-web-search-api"
        )
        return

    # Extract query parameters
    query_keywords, location_keywords = _extract_query_parameters(args.query)

    # Step 1: Discover job URLs via search API
    logger.info("Discovering jobs via search API...")
    discovered_jobs = discover_job_urls_via_search_api(
        query_keywords=query_keywords,
        location_keywords=location_keywords,
        max_results=200,
    )

    if len(discovered_jobs) == 0:
        logger.warning("Zero jobs discovered. Exiting.")
        return

    # Step 2: Filter to ATS domains only
    ats_jobs = [
        job
        for job in discovered_jobs
        if any(domain in job.get("url", "") for domain in ATS_DOMAINS)
    ]
    logger.info("Found %d ATS URLs after domain filtering", len(ats_jobs))

    if len(ats_jobs) == 0:
        logger.warning("Zero ATS URLs found. Exiting.")
        return

    # Step 3: Enrich jobs by scraping individual URLs
    logger.info("Enriching %d job URLs with full details...", len(ats_jobs))
    enriched_jobs = []
    for job in tqdm(ats_jobs, desc="Enriching jobs"):
        try:
            enriched = enrich_job_from_url(job)
            enriched_jobs.append(enriched)
        except Exception:
            logger.debug("Failed to enrich job: %s", job.get("url", ""))
            continue

    # Step 4: Filter by keywords and location
    filtered_jobs = []
    for job in enriched_jobs:
        if passes_keyword_filter(job.get("title", ""), job.get("description", "")):
            if passes_location_filter(
                job.get("location", ""), job.get("description", "")
            ):
                filtered_jobs.append(job)

    logger.info(
        "Found %d matching jobs after keyword and location filtering",
        len(filtered_jobs),
    )

    if len(filtered_jobs) == 0:
        logger.warning("Zero jobs passed keyword/location filters. Exiting.")
        return

    # Step 5: Normalize and compute delta
    normalized_df = normalize_jobs(filtered_jobs)
    master_df = load_master_csv()
    delta_df = compute_delta(master_df, normalized_df)

    # Step 6: Save CSVs
    if not delta_df.empty:
        updated_master = pd.concat([master_df, delta_df], ignore_index=True)
    else:
        updated_master = master_df

    save_csv(updated_master, MASTER_CSV_PATH)
    save_csv(delta_df, DELTA_CSV_PATH)

    logger.info("Total jobs scraped this run: %d", len(normalized_df))
    logger.info("New jobs found: %d", len(delta_df))

    print("CSV files updated successfully.")
    print(f"Total jobs scraped this run: {len(normalized_df)}")
    print(f"New jobs found: {len(delta_df)}")
    print(f"Master CSV path: {MASTER_CSV_PATH}")
    print(f"Delta CSV path: {DELTA_CSV_PATH}")


if __name__ == "__main__":
    main()
