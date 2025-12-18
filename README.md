# Weekly ATS Job Scraper

A Python script that automatically discovers and scrapes AI/engineering jobs from multiple Applicant Tracking System (ATS) platforms, filters them by location and keywords, and generates delta reports showing only new job postings.

## Problem Statement

### The Challenge

Job seekers and recruiters need to track new job postings across multiple ATS platforms (Lever, Ashby, Greenhouse, etc.) without manually checking each company's career page. The challenge is:

1. **Discovery**: Finding which companies are hiring without maintaining a static list
2. **Aggregation**: Collecting jobs from multiple ATS platforms in one place
3. **Filtering**: Focusing on relevant roles (e.g., AI/ML engineering in Europe/remote)
4. **Delta Detection**: Identifying only new jobs since the last run (avoiding duplicates)
5. **Automation**: Running weekly to catch new postings as they appear

### Why ATS Platforms?

- **Structured Data**: ATS platforms provide consistent, parseable job listings
- **No Anti-Scraping**: Unlike LinkedIn, ATS platforms don't actively block scraping
- **Direct Source**: Jobs are posted directly by companies, ensuring accuracy
- **Wide Coverage**: Many companies use ATS platforms (Lever, Ashby, Greenhouse, etc.)

## Solution Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Job Discovery Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   SerpAPI    │  │  Bing API    │  │  10 ATS      │           │
│  │  (Primary)   │  │  (Fallback)  │  │  Platforms   │           |
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           |
|         │                  │                  │                 |
|         └──────────────────┴──────────────────┘                 |
|                            │                                    |
|                    Search Results (JSON)                        |
|                            │                                    |
|                    Extract Job URLs                             |
└────────────────────────────┼────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Job Enrichment Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Lever Jobs  │  │  Ashby Jobs  │  │ Other ATS    │           │
│  │   Scraper    │  │   Scraper    │  │  Scrapers    │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                  │                  │                 │
│         └──────────────────┴──────────────────┘                 │
│                            │                                    │
│         Extract: Title, Company, Location, Description          │
└────────────────────────────┼────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Filtering Layer                              │
│  ┌──────────────┐              ┌──────────────┐                 │
│  │   Keyword    │              │  Location    │                 │
│  │   Filter     │              │   Filter     │                 │
│  │ (AI/ML/Eng)  │              │(Europe/Remote)│                │
│  └──────┬───────┘              └──────┬───────┘                 │
│         │                              │                        │
│         └──────────────┬───────────────┘                        │
│                        │                                        │
│              Matching Jobs Only                                 │
└────────────────────────┼────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Delta Detection Layer                        │
│  ┌──────────────┐              ┌──────────────┐                 │
│  │  Generate    │              │   Compare    │                 │
│  │   job_id     │              │  with Master │                 │
│  │  (SHA256)    │              │     CSV      │                 │
│  └──────┬───────┘              └──────┬───────┘                 │
│         │                              │                        │
│         └──────────────┬───────────────┘                        │
│                        │                                        │
│         New Jobs Only (Not in Master CSV)                       │
└────────────────────────┼────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Output Layer                                 │
│  ┌──────────────┐              ┌──────────────┐                 │
│  │  master_jobs │              │  delta_jobs  │                 │
│  │    .csv      │              │    .csv      │                 │
│  │ (All Jobs)   │              │ (New Jobs)   │                 │
│  └──────────────┘              └──────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

## How It Works

### 1. Job Discovery (Search API)

Instead of maintaining a static list of companies, the scraper uses search engine APIs to discover jobs dynamically:

- **SerpAPI** (primary): Google search results via API
- **Bing Web Search API** (fallback): Alternative search engine

**Search Strategy:**

- Queries each ATS platform domain with boolean search operators
- Example: `site:jobs.lever.co ("AI" OR "ML") AND ("Europe" OR "remote")`
- Searches 10 ATS platforms in parallel:
  - `jobs.lever.co`
  - `jobs.ashbyhq.com`
  - `boards.greenhouse.io`
  - `jobs.smartrecruiters.com`
  - `wd1.myworkdayjobs.com`
  - `jobs.bamboohr.com`
  - `jobs.jobvite.com`
  - `careers.icims.com`
  - `apply.jazz.co`
  - `careers.workable.com`

**Why This Approach?**

- **Automatic Discovery**: Finds companies hiring without manual maintenance
- **Comprehensive Coverage**: Searches across all major ATS platforms
- **Real-time Results**: Uses search engine indexes (up-to-date)
- **No Hardcoding**: No need to maintain company lists

### 2. Job Enrichment (Web Scraping)

For each job URL discovered, the scraper:

1. **Fetches the job page** using HTTP requests
2. **Parses HTML** using BeautifulSoup
3. **Extracts structured data**:
   - Job title
   - Company name (from URL or page)
   - Location
   - Job description (first 500 words)

**Platform-Specific Parsing:**

- **Lever**: Parses `div.posting`, `div.section-wrapper`
- **Ashby**: Parses `data-testid="job-description"` containers
- **Generic**: Falls back to common HTML patterns (`<main>`, `<article>`)

### 3. Filtering

Jobs are filtered through two criteria:

**Keyword Filter:**

- Must contain: `AI`, `Artificial Intelligence`, `ML`, `Machine Learning`, `Engineer`, `Developer`
- Searches in both title and description

**Location Filter:**

- Must mention: `remote`, `Europe`, `EU`, `UK`, or European country names
- Searches in both location field and description

### 4. Delta Detection

**Job Identity (job_id):**

- Generated using SHA256 hash of: `title + company + location + url`
- Deterministic: Same job always generates same `job_id`
- Used for deduplication across runs

**Delta Logic:**

1. Load `master_jobs.csv` (if exists)
2. Generate `job_id` for each scraped job
3. Compare against existing `job_id` values in master
4. New jobs = `job_id` not in master
5. Append new jobs to master CSV
6. Write new jobs to `delta_jobs.csv`

**Why This Works:**

- **Idempotent**: Running multiple times doesn't create duplicates
- **Persistent**: Master CSV grows over time, tracking all jobs ever seen
- **Efficient**: Only new jobs are written to delta CSV

### 5. CSV Generation

**master_jobs.csv:**

- Contains all jobs ever scraped
- Grows over time
- Used as reference for delta detection

**delta_jobs.csv:**

- Contains only new jobs from the current run
- Overwritten each run (shows latest delta)
- Used to see what's new since last run

**CSV Schema:**

```csv
job_id,title,company,location,description,url,source,scraped_at
```

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Step 1: Clone or Download

```bash
git clone <repository-url>
cd calyptus-assignment
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**

- `requests` - HTTP requests
- `beautifulsoup4` - HTML parsing
- `pandas` - CSV handling
- `python-dotenv` - Environment variable loading
- `tqdm` - Progress bars (optional)

### Step 3: Get API Key

**Option A: SerpAPI (Recommended)**

1. Sign up at https://serpapi.com/
2. Get your API key from the dashboard
3. Free tier: 100 searches/month

**Option B: Bing Web Search API**

1. Sign up at https://www.microsoft.com/en-us/bing/apis/bing-web-search-api
2. Get your subscription key
3. Free tier: 3,000 queries/month

### Step 4: Configure API Key

Create a `.env` file in the project root:

```bash
# .env
SERPAPI_KEY=your_serpapi_key_here
# OR
BING_SEARCH_API_KEY=your_bing_key_here
```

**Security Note:** Never commit `.env` to version control. It's already in `.gitignore`.

## Usage

### Basic Usage

```bash
python scraper.py "AI engineer roles in Europe or remote"
```

### Query Examples

**AI/ML Engineering Roles:**

```bash
python scraper.py "AI engineer roles in Europe or remote"
```

**Software Engineering:**

```bash
python scraper.py "Software engineer remote Europe"
```

**Data Science:**

```bash
python scraper.py "Data scientist ML engineer remote"
```

**Generic Search:**

```bash
python scraper.py "Engineering roles in Europe"
```

### Understanding the Query

The query string is used to:

1. **Document intent**: Logged for reference
2. **Extract keywords**: Automatically detects AI/ML/engineering terms
3. **Extract locations**: Automatically detects Europe/remote mentions

**Default Behavior:**

- If query contains "AI" or "ML": Uses AI/ML keywords
- If query contains "engineer" or "developer": Adds engineering keywords
- If query contains "Europe" or "remote": Uses location filters

**Customization:**
Edit `KEYWORDS` and `EUROPE_REMOTE_HINTS` in `scraper.py` to change filters.

### Output Files

After running, you'll find:

**data/master_jobs.csv**

- All jobs ever scraped
- Format: CSV with columns: `job_id`, `title`, `company`, `location`, `description`, `url`, `source`, `scraped_at`

**data/delta_jobs.csv**

- Only new jobs from the current run
- Same format as master CSV
- Empty if no new jobs found

### Running Weekly

**Manual:**

```bash
# Run every Monday
python scraper.py "AI engineer roles in Europe or remote"
```

**Automated (Cron/Scheduled Task):**

```bash
# Add to crontab (Linux/Mac)
0 9 * * 1 cd /path/to/project && python scraper.py "AI engineer roles in Europe or remote"

# Windows Task Scheduler
# Create task to run: python C:\path\to\scraper.py "AI engineer roles in Europe or remote"
```

## Example Output

### Console Output

```
2025-12-18 21:21:24,009 [INFO] Starting ATS job scrape...
2025-12-18 21:21:24,009 [INFO] Search query (for documentation/intention): AI engineer roles in Europe or remote
2025-12-18 21:21:24,009 [INFO] Discovering jobs via search API (SerpAPI or Bing API)...
2025-12-18 21:21:24,009 [INFO] Searching for jobs via search API...
2025-12-18 21:21:24,009 [INFO] Searching jobs.lever.co with query: site:jobs.lever.co ("AI" OR "Artificial Intelligence" OR "ML") AND ("Europe" OR "EU" OR "UK" OR "remote")
2025-12-18 21:21:25,893 [INFO] SerpAPI returned 10 results for jobs.lever.co
2025-12-18 21:21:25,893 [INFO] Found 10 job URLs from jobs.lever.co via SerpAPI
...
2025-12-18 21:21:45,627 [INFO] Found 67 total job URLs via search API
2025-12-18 21:21:45,629 [INFO] Found 67 ATS URLs after filtering
2025-12-18 21:21:45,629 [INFO] Enriching 67 job URLs with full details...
Enriching jobs: 100%|##########| 67/67 [02:15<00:00,  2.02s/it]
2025-12-18 21:24:00,688 [INFO] Found 32 matching jobs after enrichment and filtering
CSV files updated successfully.
Total jobs scraped this run: 32
New jobs found: 28
Master CSV path: data\master_jobs.csv
Delta CSV path: data\delta_jobs.csv
```

### CSV Sample

**master_jobs.csv:**

```csv
"job_id","title","company","location","description","url","source","scraped_at"
"62fe0ceb77256d87c03737825c3c9645e8fd4e27f28a9517dc2f65f626272c26","Tech Lead, LLM & Generative AI (Full Remote - Europe)","jobgether","European Economic AreaSecurity & IT – IT /Full-time /Remote","","https://jobs.lever.co/jobgether/e058e92e-c57f-45f2-bf81-cec5d3117361","lever","2025-12-18T15:28:20.936699+00:00"
```

**delta_jobs.csv:**

```csv
"job_id","title","company","location","description","url","source","scraped_at"
"d23dc316872203eb7c680b7ac3079bf643d16dbab311e4108c1f97f2c211790","Jobgether - Staff Product Manager, AI [gn] Europe","jobgether","European Economic AreaResearch & Development – Product /Full-time /Remote","","https://jobs.lever.co/jobgether/4b75fa12-a30f-4d63-9922-add837da8d46","lever","2025-12-18T15:54:00.688355+00:00"
```

## Architecture Details

### Key Design Decisions

1. **Search API Over HTML Scraping**

   - Avoids CAPTCHA/rate limiting
   - More reliable than parsing HTML
   - Respects platform terms of service

2. **Deterministic Job IDs**

   - SHA256 hash ensures consistency
   - Same job always gets same ID
   - Enables reliable delta detection

3. **Idempotent Design**

   - Can run multiple times safely
   - Master CSV grows, delta CSV shows new only
   - No duplicate jobs in master

4. **Fault Tolerance**

   - Continues if individual jobs fail
   - Logs errors but doesn't crash
   - Handles 403/404 gracefully

5. **Platform Agnostic**
   - Easy to add new ATS platforms
   - Just add domain to `ATS_DOMAINS` list
   - Search API automatically discovers jobs

### File Structure

```
calyptus-assignment/
├── scraper.py              # Main script
├── .env                    # API keys (not in git)
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── data/
    ├── master_jobs.csv    # All jobs ever seen
    └── delta_jobs.csv     # New jobs from last run
```

## Troubleshooting

### No Jobs Found

**Possible Causes:**

1. **API Key Missing**: Check `.env` file exists and has correct key
2. **API Quota Exceeded**: Check API dashboard for remaining quota
3. **No Matching Jobs**: Try broader search terms
4. **Network Issues**: Check internet connection

**Solutions:**

```bash
# Verify API key is loaded
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('SERPAPI_KEY:', 'SET' if os.getenv('SERPAPI_KEY') else 'NOT SET')"

# Check API quota
# Visit SerpAPI dashboard: https://serpapi.com/dashboard
# Or Bing API portal: https://portal.azure.com
```

### Zero New Jobs in Delta

**This is Normal!**

- If you run the scraper twice quickly, delta will be empty
- Delta only shows jobs not in master CSV
- Wait for new jobs to be posted, then run again

### Errors During Enrichment

**Common Issues:**

- **403 Forbidden**: Job page blocks scraping (logged as warning, continues)
- **404 Not Found**: Job was removed (logged as warning, continues)
- **Timeout**: Job page takes too long (logged, continues)

**These are Expected:**

- The scraper handles these gracefully
- Failed jobs are skipped, successful ones are saved
- Check logs for details

### API Rate Limits

**SerpAPI:**

- Free tier: 100 searches/month
- Paid tiers available

**Bing API:**

- Free tier: 3,000 queries/month
- Paid tiers available

**Mitigation:**

- Script includes 1-second delay between platform searches
- Consider upgrading API tier for more searches
- Run less frequently (weekly instead of daily)

## Extending the Scraper

### Adding New ATS Platforms

1. Add domain to `ATS_DOMAINS` list in `scraper.py`:

```python
ATS_DOMAINS = [
    "jobs.lever.co",
    "jobs.ashbyhq.com",
    "your-new-platform.com",  # Add here
]
```

2. Add platform to `ats_platforms` in `search_jobs_via_api()`:

```python
ats_platforms = [
    ("jobs.lever.co", "lever"),
    ("your-new-platform.com", "newplatform"),  # Add here
]
```

3. Add parsing logic in `enrich_job_from_url()` if needed:

```python
elif source == "newplatform":
    # Add parsing logic here
    pass
```

### Changing Filters

**Keywords:**
Edit `KEYWORDS` list in `scraper.py`:

```python
KEYWORDS = [
    "ai",
    "machine learning",
    "your-keyword",  # Add here
]
```

**Locations:**
Edit `EUROPE_REMOTE_HINTS` list:

```python
EUROPE_REMOTE_HINTS = [
    "remote",
    "europe",
    "your-location",  # Add here
]
```

### Adding Slack/Email Notifications

**Slack Webhook Example:**

```python
import requests

def send_to_slack(delta_df):
    if delta_df.empty:
        return

    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    message = f"Found {len(delta_df)} new jobs!"

    requests.post(webhook_url, json={"text": message})
```

**Email Example:**

```python
import smtplib
from email.mime.text import MIMEText

def send_email(delta_df):
    if delta_df.empty:
        return

    # Add email sending logic here
    pass
```

## Performance

### Typical Run Time

- **Discovery**: 20-30 seconds (10 API calls)
- **Enrichment**: 2-3 minutes (67 job pages)
- **Total**: ~3 minutes for 67 jobs

### Scalability

- **Current**: Handles ~100 jobs per run efficiently
- **Limitations**: API quotas, network speed
- **Optimization**: Parallel enrichment possible (not implemented)

## Security & Privacy

- **API Keys**: Stored in `.env` (not committed to git)
- **No Personal Data**: Only scrapes public job postings
- **Rate Limiting**: Respects API rate limits
- **User-Agent**: Identifies as ATSJobScraper/1.0

## License

This project is provided as-is for educational/demonstration purposes.

## Contributing

Feel free to submit issues or pull requests for improvements.

## Support

For issues or questions:

1. Check the Troubleshooting section
2. Review logs for error messages
3. Verify API keys and quotas
4. Check network connectivity

---

**Built with:** Python, SerpAPI, BeautifulSoup, Pandas

**Last Updated:** December 2024
