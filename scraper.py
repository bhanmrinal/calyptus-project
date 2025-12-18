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

    load_dotenv()  # Load environment variables from .env file
except ImportError:
    # python-dotenv not installed, skip loading .env
    pass

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - very small fallback

    def tqdm(iterable: Iterable, **_: Dict) -> Iterable:
        return iterable


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


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

# Allowed ATS domains for job discovery
ATS_DOMAINS = [
    "jobs.lever.co",
    "jobs.ashbyhq.com",
    "boards.greenhouse.io",
    "jobs.smartrecruiters.com",
    "wd1.myworkdayjobs.com",
    "jobs.bamboohr.com",
    "jobs.jobvite.com",
    "careers.icims.com",
    "apply.jazz.co",
    "careers.workable.com",
]

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ATSJobScraper/1.0; +https://example.com)",
    "Accept-Language": "en-US,en;q=0.9",
}


def load_env_from_file(path: Path = Path(".env")) -> None:
    """
    Very small .env loader so we don't depend on external libraries.
    Loads KEY=VALUE lines into os.environ if not already set.
    """
    try:
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        # Safe to ignore .env failures; environment variables still work.
        logger.debug("Failed to load environment variables from %s", path)


# Load environment variables from .env if present
load_env_from_file()


def generate_job_id(title: str, company: str, location: str, url: str) -> str:
    """Generate a deterministic job ID based on title, company, location, and URL."""
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
    """Load the master CSV if it exists, otherwise return an empty DataFrame."""
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
    """Save a DataFrame to CSV, ensuring the parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL)


def fetch_url(url: str) -> Optional[str]:
    """Fetch URL content with basic error handling."""
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        if response.status_code != 200:
            logger.warning("Non-200 status %s for URL %s", response.status_code, url)
            return None
        return response.text
    except requests.RequestException:
        logger.exception("Request failed for URL %s", url)
        return None


def passes_keyword_filter(title: str, description: str) -> bool:
    """Return True if the job matches any of the keyword filters."""
    combined = f"{title} {description}".lower()
    return any(keyword in combined for keyword in KEYWORDS)


def passes_location_filter(location: str, description: str) -> bool:
    """
    Simple Europe / Remote heuristic.

    - If location is empty, fall back to description text.
    - Keep jobs mentioning 'remote', 'europe', or common European country names.
    """
    text = f"{location} {description}".lower()
    return any(hint in text for hint in EUROPE_REMOTE_HINTS)


def search_jobs_via_api(
    query_keywords: List[str], location_keywords: List[str], max_results: int = 100
) -> List[Dict[str, str]]:
    """
    Search for jobs using search API (SerpAPI or Bing Web Search API).
    Returns a list of job dicts with url, title, company, location, description, source.

    Discovery is performed via search-engine APIs indexing ATS domains,
    rather than static company lists, to ensure coverage of newly hiring companies.
    """
    logger.info("Searching for jobs via search API...")

    # Map ATS domains to source names
    ats_platforms = [
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

    all_jobs: List[Dict[str, str]] = []
    seen_urls = set()

    # Build simple queries respecting Google's 32-word limit
    # Use small keyword/location sets to keep queries manageable
    limited_keywords = query_keywords[:3]  # Limit to 3 keywords
    limited_locations = location_keywords[:4]  # Limit to 4 locations

    keyword_part = " OR ".join([f'"{kw}"' for kw in limited_keywords])
    location_part = " OR ".join([f'"{loc}"' for loc in limited_locations])

    # Get API keys once at the start
    serpapi_key = os.getenv("SERPAPI_KEY")
    bing_api_key = os.getenv("BING_SEARCH_API_KEY")

    if not serpapi_key and not bing_api_key:
        logger.error(
            "No search API key found. Set SERPAPI_KEY or BING_SEARCH_API_KEY environment variable."
        )
        return []

    for platform_domain, source_name in ats_platforms:
        try:
            # Build search query: site:platform (keywords) AND (locations)
            # Keep simple to respect 32-word limit
            search_query = (
                f"site:{platform_domain} ({keyword_part}) AND ({location_part})"
            )

            logger.info("Searching %s with query: %s", platform_domain, search_query)

            # Try SerpAPI first, fallback to Bing Web Search API
            urls_found = []
            api_used = None

            # Try SerpAPI (recommended)
            if serpapi_key:
                try:
                    serpapi_url = "https://serpapi.com/search"
                    params = {
                        "q": search_query,
                        "api_key": serpapi_key,
                        "num": 50,
                        "engine": "google",
                    }
                    response = requests.get(serpapi_url, params=params, timeout=20)
                    if response.status_code == 200:
                        data = response.json()
                        api_used = "SerpAPI"
                        # Extract URLs from SerpAPI JSON response
                        if "organic_results" in data:
                            for result in data["organic_results"]:
                                url = result.get("link", "")
                                title = result.get("title", "")
                                if url and platform_domain in url:
                                    urls_found.append((url, title))
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
                    logger.debug(
                        "SerpAPI request failed for %s: %s", platform_domain, e
                    )
                except Exception as e:
                    logger.debug(
                        "SerpAPI parsing failed for %s: %s", platform_domain, e
                    )

            # Fallback to Bing Web Search API if SerpAPI failed or not configured
            if not urls_found and bing_api_key:
                try:
                    bing_url = "https://api.bing.microsoft.com/v7.0/search"
                    headers = {"Ocp-Apim-Subscription-Key": bing_api_key}
                    params = {"q": search_query, "count": 50}
                    response = requests.get(
                        bing_url, headers=headers, params=params, timeout=20
                    )
                    if response.status_code == 200:
                        data = response.json()
                        api_used = "Bing API"
                        # Extract URLs from Bing API JSON response
                        if "webPages" in data and "value" in data["webPages"]:
                            for result in data["webPages"]["value"]:
                                url = result.get("url", "")
                                title = result.get("name", "")
                                if url and platform_domain in url:
                                    urls_found.append((url, title))
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
                    logger.debug(
                        "Bing API request failed for %s: %s", platform_domain, e
                    )
                except Exception as e:
                    logger.debug(
                        "Bing API parsing failed for %s: %s", platform_domain, e
                    )

            # Process found URLs
            found_count = 0
            for url, title in urls_found:
                if url in seen_urls:
                    continue

                # Filter out search pages, not actual job postings
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

                # Create initial job dict (will be enriched by scraping)
                job = {
                    "url": url,
                    "title": title or "Job Posting",
                    "company": company,
                    "location": "",
                    "description": "",
                    "source": source_name,
                }
                all_jobs.append(job)
                found_count += 1

            if found_count > 0:
                logger.info(
                    "Found %d job URLs from %s via %s",
                    found_count,
                    platform_domain,
                    api_used or "Unknown",
                )
            else:
                logger.debug("No job URLs found for %s", platform_domain)

            # Rate limiting
            time.sleep(1)

        except Exception:
            logger.exception("Error searching %s", platform_domain)
            continue

    logger.info("Found %d total job URLs via search API", len(all_jobs))
    if len(all_jobs) == 0:
        logger.warning(
            "Zero jobs found. Check API keys and quotas. Set SERPAPI_KEY or BING_SEARCH_API_KEY environment variable."
        )
    return all_jobs[:max_results]


def scrape_lever_company(company: str) -> List[Dict[str, str]]:
    """
    Scrape jobs from a single Lever company page.
    Returns a list of normalized job dicts without job_id/scraped_at.
    """
    base_url = f"https://jobs.lever.co/{company}"
    html = fetch_url(base_url)
    if not html:
        logger.warning("No HTML returned for Lever company %s", company)
        return []

    soup = BeautifulSoup(html, "html.parser")
    postings = soup.find_all("div", class_="posting")

    jobs: List[Dict[str, str]] = []

    for posting in postings:
        link = posting.find("a", href=True)
        if not link:
            continue

        title_el = posting.find("h5") or linking_text_element(link=link)
        title = title_el.get_text(strip=True) if title_el else link.get_text(strip=True)
        location_el = posting.find("span", class_="sort-by-location")
        location = location_el.get_text(strip=True) if location_el else ""

        job_url = link["href"]
        if not job_url.startswith("http"):
            job_url = f"https://jobs.lever.co{job_url}"

        description = fetch_lever_job_description(job_url)

        if not passes_keyword_filter(title, description):
            continue
        if not passes_location_filter(location, description):
            continue

        jobs.append(
            {
                "title": title,
                "company": company,
                "location": location or "Unknown",
                "description": description,
                "url": job_url,
                "source": "lever",
            }
        )

    return jobs


def linking_text_element(link) -> Optional[BeautifulSoup]:
    """
    Fallback helper to get a title element from a link if structured tags are missing.
    """
    return link


def fetch_lever_job_description(job_url: str) -> str:
    """Fetch basic job description text from Lever job detail page."""
    html = fetch_url(job_url)
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    description_container = soup.find("div", class_="section-wrapper")
    if not description_container:
        description_container = soup.find("div", class_="posting")
    if not description_container:
        return ""
    return " ".join(description_container.get_text(separator=" ", strip=True).split())


def scrape_lever() -> List[Dict[str, str]]:
    """Backward-compatible wrapper that scrapes Lever using default companies."""
    default_companies = [
        "openai",
        "huggingface",
        "deepmind",
        "stabilityai",
    ]
    return scrape_lever_for_companies(default_companies)


def scrape_lever_for_companies(companies: List[str]) -> List[Dict[str, str]]:
    """Scrape Lever for the given companies, logging per-company counts."""
    all_jobs: List[Dict[str, str]] = []
    for company in tqdm(companies, desc="Lever companies"):
        try:
            company_jobs = scrape_lever_company(company)
        except Exception:
            logger.exception("Failed to scrape Lever for company %s", company)
            continue

        logger.info("[lever] %s: scraped %d matching jobs", company, len(company_jobs))
        all_jobs.extend(company_jobs)
    return all_jobs


def scrape_ashby_company(company: str) -> List[Dict[str, str]]:
    """
    Scrape jobs from a single Ashby company page.
    Returns a list of normalized job dicts without job_id/scraped_at.
    """
    base_url = f"https://jobs.ashbyhq.com/{company}"
    html = fetch_url(base_url)
    if not html:
        logger.warning("No HTML returned for Ashby company %s", company)
        return []

    soup = BeautifulSoup(html, "html.parser")
    jobs: List[Dict[str, str]] = []
    seen_links = set()

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/job/" not in href:
            continue
        if href.startswith("mailto:") or href.startswith("tel:"):
            continue
        if href in seen_links:
            continue
        seen_links.add(href)

        job_url = href
        if not job_url.startswith("http"):
            if job_url.startswith("/"):
                job_url = f"https://jobs.ashbyhq.com{job_url}"
            else:
                job_url = f"https://jobs.ashbyhq.com/{href}"

        title = link.get_text(strip=True)
        if not title:
            continue

        # Default Ashby location assumption for this simple implementation
        location = "Remote / Europe"
        description = fetch_ashby_job_description(job_url)

        if not passes_keyword_filter(title, description):
            continue
        if not passes_location_filter(location, description):
            continue

        jobs.append(
            {
                "title": title,
                "company": company,
                "location": location,
                "description": description,
                "url": job_url,
                "source": "ashby",
            }
        )

    return jobs


def fetch_ashby_job_description(job_url: str) -> str:
    """Fetch basic job description text from Ashby job detail page."""
    html = fetch_url(job_url)
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    # Try common Ashby containers; fall back to full page text if needed.
    description_container = (
        soup.find("section", attrs={"data-testid": "job-description"})
        or soup.find("div", attrs={"data-testid": "job-description"})
        or soup.find("main")
    )
    if not description_container:
        return ""

    return " ".join(description_container.get_text(separator=" ", strip=True).split())


def scrape_ashby() -> List[Dict[str, str]]:
    """Backward-compatible wrapper that scrapes Ashby using default companies."""
    default_companies = [
        "perplexity",
        "mistral",
        "anthropic",
        "cohere",
    ]
    return scrape_ashby_for_companies(default_companies)


def scrape_ashby_for_companies(companies: List[str]) -> List[Dict[str, str]]:
    """Scrape Ashby for the given companies, logging per-company counts."""
    all_jobs: List[Dict[str, str]] = []
    for company in tqdm(companies, desc="Ashby companies"):
        try:
            company_jobs = scrape_ashby_company(company)
        except Exception:
            logger.exception("Failed to scrape Ashby for company %s", company)
            continue

        logger.info("[ashby] %s: scraped %d matching jobs", company, len(company_jobs))
        all_jobs.extend(company_jobs)
    return all_jobs


def enrich_job_from_url(job: Dict[str, str]) -> Dict[str, str]:
    """
    Enrich a job dict (found via search) by scraping the actual job URL
    to get full title, location, description, and company details.
    """
    url = job.get("url", "")
    if not url or not url.startswith("http"):
        return job

    source = job.get("source", "")
    html = fetch_url(url)
    if not html:
        return job

    soup = BeautifulSoup(html, "html.parser")

    # Lever-specific parsing
    if source == "lever":
        # Try to get better title
        title_el = soup.find("h2", class_="posting-headline") or soup.find("h1")
        if title_el:
            job["title"] = title_el.get_text(strip=True)

        # Try to get location
        location_el = soup.find("div", class_="posting-categories") or soup.find(
            "span", class_="sort-by-location"
        )
        if location_el:
            job["location"] = location_el.get_text(strip=True)

        # Get description
        desc_el = soup.find("div", class_="section-wrapper") or soup.find(
            "div", class_="posting"
        )
        if desc_el:
            job["description"] = " ".join(
                desc_el.get_text(separator=" ", strip=True).split()[:500]
            )  # Limit length

        # Extract company from URL if not set
        if job.get("company") == "Unknown":
            match = re.search(r"jobs\.lever\.co/([^/]+)", url)
            if match:
                job["company"] = match.group(1)

    # Ashby-specific parsing
    elif source == "ashby":
        # Try to get better title
        title_el = soup.find("h1") or soup.find("title")
        if title_el:
            job["title"] = title_el.get_text(strip=True)

        # Try to get location
        location_el = soup.find(
            "div", attrs={"data-testid": "job-location"}
        ) or soup.find("span", class_="location")
        if location_el:
            job["location"] = location_el.get_text(strip=True)

        # Get description
        desc_el = (
            soup.find("section", attrs={"data-testid": "job-description"})
            or soup.find("div", attrs={"data-testid": "job-description"})
            or soup.find("main")
        )
        if desc_el:
            job["description"] = " ".join(
                desc_el.get_text(separator=" ", strip=True).split()[:500]
            )

        # Extract company from URL if not set
        if job.get("company") == "Unknown":
            match = re.search(r"jobs\.ashbyhq\.com/([^/]+)", url)
            if match:
                job["company"] = match.group(1)

    # Generic fallback for other ATS platforms
    else:
        title_el = soup.find("h1") or soup.find("title")
        if title_el:
            job["title"] = title_el.get_text(strip=True)

        # Try to find description in common containers
        desc_el = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", class_="description")
        )
        if desc_el:
            job["description"] = " ".join(
                desc_el.get_text(separator=" ", strip=True).split()[:500]
            )

    return job


def normalize_jobs(raw_jobs: List[Dict[str, str]]) -> pd.DataFrame:
    """Normalize raw job dicts into a DataFrame with required schema."""
    now_iso = datetime.now(UTC).isoformat()

    normalized: List[Dict[str, str]] = []
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


def compute_delta(master_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    """Compute new jobs by job_id compared to master."""
    if master_df.empty:
        return current_df.copy()

    master_ids = set(master_df["job_id"].astype(str))
    mask = ~current_df["job_id"].astype(str).isin(master_ids)
    return current_df[mask].copy()


def main() -> None:
    """Main entry point for CLI execution."""
    parser = argparse.ArgumentParser(
        description="Weekly ATS Job Scraper using search API discovery."
    )
    parser.add_argument(
        "query",
        type=str,
        help=(
            "Free-form search query description "
            '(e.g. "AI engineer roles in Europe / remote"). '
            "Used for logging and to document the intent of this run."
        ),
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting ATS job scrape...")
    logger.info("Search query (for documentation/intention): %s", args.query)

    # Check for API keys
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

    # Extract keywords and location hints from query
    # Default to AI/engineering keywords and Europe/remote locations
    query_keywords = KEYWORDS.copy()
    location_keywords = EUROPE_REMOTE_HINTS.copy()

    # Try to extract from user query (simple heuristic)
    query_lower = args.query.lower()
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

    # Search for jobs via API
    logger.info("Discovering jobs via search API (SerpAPI or Bing API)...")
    search_jobs = search_jobs_via_api(
        query_keywords=query_keywords,
        location_keywords=location_keywords,
        max_results=200,
    )

    logger.info("Found %d job URLs from search API", len(search_jobs))

    # Filter URLs to ATS domains only
    ats_jobs = []
    for job in search_jobs:
        url = job.get("url", "")
        if any(domain in url for domain in ATS_DOMAINS):
            ats_jobs.append(job)

    logger.info("Found %d ATS URLs after filtering", len(ats_jobs))

    if len(ats_jobs) == 0:
        logger.warning("Zero jobs found. Check API keys, quotas, and search queries.")
        return

    # Enrich jobs with full details by scraping each URL
    logger.info("Enriching %d job URLs with full details...", len(ats_jobs))
    enriched_jobs = []
    for job in tqdm(ats_jobs, desc="Enriching jobs"):
        try:
            enriched = enrich_job_from_url(job)
            # Apply filters
            if passes_keyword_filter(
                enriched.get("title", ""), enriched.get("description", "")
            ):
                if passes_location_filter(
                    enriched.get("location", ""), enriched.get("description", "")
                ):
                    enriched_jobs.append(enriched)
        except Exception:
            logger.debug("Failed to enrich job: %s", job.get("url", ""))
            continue

    logger.info(
        "Found %d matching jobs after enrichment and filtering", len(enriched_jobs)
    )
    all_raw_jobs = enriched_jobs
    all_jobs_df = normalize_jobs(all_raw_jobs)

    master_df = load_master_csv()
    delta_df = compute_delta(master_df, all_jobs_df)

    # Append new jobs to master and save both CSVs
    if not delta_df.empty:
        updated_master = pd.concat([master_df, delta_df], ignore_index=True)
    else:
        updated_master = master_df

    total_scraped = len(all_jobs_df)
    new_jobs_count = len(delta_df)

    logger.info("Total jobs scraped this run (after filters): %d", total_scraped)
    logger.info("New jobs found vs master: %d", new_jobs_count)

    save_csv(updated_master, MASTER_CSV_PATH)
    save_csv(delta_df, DELTA_CSV_PATH)

    print("CSV files updated successfully.")
    print(f"Total jobs scraped this run: {total_scraped}")
    print(f"New jobs found: {new_jobs_count}")
    print(f"Master CSV path: {MASTER_CSV_PATH}")
    print(f"Delta CSV path: {DELTA_CSV_PATH}")


if __name__ == "__main__":
    main()
