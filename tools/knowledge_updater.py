"""
knowledge_updater.py — Research paper crawl pipeline for partner-decode-female-agent.

Sources: ArXiv (cs.CL, cs.CV, cs.HC) + Semantic Scholar Graph API
Schedule: Weekly, Sunday 02:00 local time
Output: Appends new papers to SECOND-KNOWLEDGE-BRAIN.md
Deduplication: SHA256 hash of paper URL/DOI stored in SQLite
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

ARXIV_QUERIES = [
    "female partner communication emotion",
    "attachment theory NLP detection",
    "Gottman communication patterns",
    "multimodal emotion recognition speech",
    "facial action unit relationship",
]

S2_QUERIES = [
    "female partner communication emotion analysis",
    "attachment theory behavioral markers NLP",
    "gottman communication patterns machine learning",
    "multimodal emotion recognition speech text",
    "facial action unit relationship communication",
]

ARXIV_CATEGORIES = ["cs.CL", "cs.CV", "cs.HC"]

DOMAIN_KEYWORDS = [
    "emotion recognition",
    "attachment theory",
    "gottman",
    "communication patterns",
    "facial expression",
    "speech prosody",
    "relationship quality",
    "affective computing",
    "sentiment analysis",
    "love language",
    "multimodal",
    "couple communication",
    "hedging language",
    "non-verbal cues",
    "partner behavior",
]

BRAIN_PATH = Path(__file__).parent.parent / "SECOND-KNOWLEDGE-BRAIN.md"


@dataclass
class PaperEntry:
    """Represents a single research paper fetched from any source."""

    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: int = 0
    venue: str = ""
    url: str = ""
    doi: str = ""
    abstract: str = ""
    key_finding: str = ""
    relevance: str = ""
    source: str = ""  # "arxiv" | "semantic_scholar"
    published_date: Optional[datetime] = None
    relevance_score: float = 0.0

    @property
    def url_hash(self) -> str:
        identifier = self.doi or self.url or self.title
        return hashlib.sha256(identifier.encode()).hexdigest()


class KnowledgeUpdater:
    """
    Crawls ArXiv and Semantic Scholar for the latest relationship science
    and affective computing papers, scores them, deduplicates, and appends
    to SECOND-KNOWLEDGE-BRAIN.md.
    """

    def __init__(
        self,
        memory_manager=None,
        max_papers_per_run: int = 50,
        brain_path: Optional[Path] = None,
        request_timeout: int = 30,
    ) -> None:
        self._memory = memory_manager
        self.max_papers_per_run = max_papers_per_run
        self.brain_path = brain_path or BRAIN_PATH
        self.request_timeout = request_timeout

    def run(self) -> int:
        """
        Execute the full crawl pipeline.
        Returns the number of new papers added.
        """
        logger.info("KnowledgeUpdater: starting crawl run")
        all_papers: List[PaperEntry] = []

        # Fetch from ArXiv
        for query in ARXIV_QUERIES:
            for category in ARXIV_CATEGORIES:
                try:
                    papers = self._fetch_arxiv(query, category, max_results=10)
                    all_papers.extend(papers)
                    time.sleep(3)  # ArXiv rate limit courtesy
                except Exception as exc:
                    logger.warning("ArXiv fetch failed (query=%s, cat=%s): %s", query, category, exc)

        # Fetch from Semantic Scholar
        for query in S2_QUERIES:
            try:
                papers = self._fetch_semantic_scholar(query, max_results=10)
                all_papers.extend(papers)
                time.sleep(1)
            except Exception as exc:
                logger.warning("S2 fetch failed (query=%s): %s", query, exc)

        if not all_papers:
            logger.info("No papers fetched this run")
            return 0

        # Deduplicate
        new_papers = self._deduplicate(all_papers)
        if not new_papers:
            logger.info("All %d papers already in knowledge base", len(all_papers))
            return 0

        # Score and rank
        scored = self._score_papers(new_papers)
        top_papers = scored[: self.max_papers_per_run]

        # Append to brain
        self._append_to_brain(top_papers)
        self._log_update(len(top_papers))

        # Store hashes
        if self._memory:
            for paper in top_papers:
                self._memory.add_knowledge_hash(paper.doi or paper.url or paper.title)

        logger.info("KnowledgeUpdater: added %d new papers", len(top_papers))
        return len(top_papers)

    def start_scheduler(self) -> None:
        """Schedule weekly Sunday 02:00 runs via APScheduler."""
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger

            scheduler = BackgroundScheduler()
            scheduler.add_job(
                func=self.run,
                trigger=CronTrigger(day_of_week="sun", hour=2, minute=0),
                id="knowledge_update",
                name="Weekly knowledge base update",
                replace_existing=True,
            )
            scheduler.start()
            logger.info("KnowledgeUpdater scheduler started: weekly Sunday 02:00")
        except ImportError:
            logger.warning("APScheduler not installed — run knowledge_updater.run() manually")

    # ------------------------------------------------------------------
    # Fetch Methods
    # ------------------------------------------------------------------

    def _fetch_arxiv(
        self, query: str, category: str, max_results: int = 10
    ) -> List[PaperEntry]:
        """Fetch papers from ArXiv API and parse XML response."""
        search_query = f"cat:{category} AND all:{query}"
        params = {
            "search_query": search_query,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = requests.get(ARXIV_API_URL, params=params, timeout=self.request_timeout)
        resp.raise_for_status()
        return self._parse_arxiv_xml(resp.text)

    def _parse_arxiv_xml(self, xml_text: str) -> List[PaperEntry]:
        """Parse ArXiv Atom XML feed into PaperEntry list."""
        papers: List[PaperEntry] = []
        try:
            root = ET.fromstring(xml_text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                summary_el = entry.find("atom:summary", ns)
                published_el = entry.find("atom:published", ns)
                id_el = entry.find("atom:id", ns)
                authors = [
                    a.find("atom:name", ns).text
                    for a in entry.findall("atom:author", ns)
                    if a.find("atom:name", ns) is not None
                ]
                title = title_el.text.strip().replace("\n", " ") if title_el is not None else ""
                abstract = summary_el.text.strip().replace("\n", " ") if summary_el is not None else ""
                arxiv_id = id_el.text.strip() if id_el is not None else ""
                published_date = None
                year = 0
                if published_el is not None:
                    try:
                        published_date = datetime.fromisoformat(
                            published_el.text.strip().replace("Z", "+00:00")
                        )
                        year = published_date.year
                    except Exception:
                        pass
                papers.append(PaperEntry(
                    title=title,
                    authors=authors[:5],
                    year=year,
                    venue="arXiv",
                    url=arxiv_id,
                    abstract=abstract,
                    key_finding=abstract[:200] + "..." if len(abstract) > 200 else abstract,
                    source="arxiv",
                    published_date=published_date,
                ))
        except ET.ParseError as exc:
            logger.error("ArXiv XML parse error: %s", exc)
        return papers

    def _fetch_semantic_scholar(
        self, query: str, max_results: int = 10
    ) -> List[PaperEntry]:
        """Fetch papers from Semantic Scholar Graph API."""
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,year,venue,externalIds,abstract,publicationDate",
        }
        resp = requests.get(S2_SEARCH_URL, params=params, timeout=self.request_timeout)
        if resp.status_code == 429:
            logger.warning("Semantic Scholar rate limited — skipping")
            return []
        resp.raise_for_status()
        data = resp.json()
        return [self._s2_to_paper_entry(item) for item in data.get("data", [])]

    def _s2_to_paper_entry(self, item: dict) -> PaperEntry:
        """Convert Semantic Scholar API item to PaperEntry."""
        doi = (item.get("externalIds") or {}).get("DOI", "")
        url = f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}"
        authors = [a.get("name", "") for a in item.get("authors", [])[:5]]
        abstract = item.get("abstract") or ""
        published_date = None
        year = item.get("year") or 0
        if item.get("publicationDate"):
            try:
                published_date = datetime.fromisoformat(item["publicationDate"])
            except Exception:
                pass
        return PaperEntry(
            title=item.get("title", ""),
            authors=authors,
            year=year,
            venue=item.get("venue", "Semantic Scholar"),
            url=url,
            doi=doi,
            abstract=abstract,
            key_finding=abstract[:200] + "..." if len(abstract) > 200 else abstract,
            source="semantic_scholar",
            published_date=published_date,
        )

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate(self, papers: List[PaperEntry]) -> List[PaperEntry]:
        """
        Remove papers already in the knowledge base (SHA256 hash check).
        Also dedup within the current batch.
        """
        seen_hashes = set()
        unique = []
        for paper in papers:
            h = paper.url_hash
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            # Check database
            if self._memory and self._memory.has_knowledge_hash(
                paper.doi or paper.url or paper.title
            ):
                continue
            unique.append(paper)
        return unique

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_papers(self, papers: List[PaperEntry]) -> List[PaperEntry]:
        """
        Score papers by: recency × 0.6 + relevance × 0.4
        Sort descending by score.
        """
        now = datetime.now()
        for paper in papers:
            recency = 0.0
            if paper.published_date:
                days_old = (now - paper.published_date.replace(tzinfo=None)).days
                if days_old <= 30:
                    recency = 1.0
                elif days_old <= 90:
                    recency = 0.8
                elif days_old <= 365:
                    recency = 0.5
                else:
                    recency = max(0.0, 1.0 - (days_old / 730))
            elif paper.year:
                years_old = now.year - paper.year
                recency = max(0.0, 1.0 - (years_old * 0.15))

            # Relevance: keyword match in title + abstract
            combined = (paper.title + " " + paper.abstract).lower()
            matches = sum(1 for kw in DOMAIN_KEYWORDS if kw in combined)
            relevance = min(1.0, matches / 5.0)

            paper.relevance_score = recency * 0.6 + relevance * 0.4

        papers.sort(key=lambda p: p.relevance_score, reverse=True)
        return papers

    # ------------------------------------------------------------------
    # Knowledge Brain Update
    # ------------------------------------------------------------------

    def _append_to_brain(self, papers: List[PaperEntry]) -> None:
        """Append new papers to the Key Research Papers table in SECOND-KNOWLEDGE-BRAIN.md."""
        if not papers:
            return
        if not self.brain_path.exists():
            logger.warning("SECOND-KNOWLEDGE-BRAIN.md not found at %s", self.brain_path)
            return

        content = self.brain_path.read_text(encoding="utf-8")

        # Build new table rows
        new_rows = []
        for paper in papers:
            authors_str = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors_str += " et al."
            key_finding = re.sub(r"\s+", " ", paper.key_finding[:150])
            url = paper.doi if paper.doi else paper.url
            row = (
                f"| (new) | {paper.title[:60]} | {authors_str} | {paper.year} | "
                f"{paper.venue[:20]} | {url[:60]} | {key_finding} | "
                f"Added by crawler (score={paper.relevance_score:.2f}) |"
            )
            new_rows.append(row)

        # Insert before ## State-of-the-Art Models
        insert_marker = "## State-of-the-Art Models"
        if insert_marker in content:
            insert_point = content.index(insert_marker)
            new_section = "\n".join(new_rows) + "\n\n"
            content = content[:insert_point] + new_section + content[insert_point:]
        else:
            content += "\n\n" + "\n".join(new_rows) + "\n"

        self.brain_path.write_text(content, encoding="utf-8")
        logger.info("Appended %d papers to SECOND-KNOWLEDGE-BRAIN.md", len(papers))

    def _log_update(self, count: int) -> None:
        """Append an entry to the Knowledge Update Log section."""
        if not self.brain_path.exists():
            return
        content = self.brain_path.read_text(encoding="utf-8")
        today = datetime.now().strftime("%Y-%m-%d")
        new_log_row = (
            f"| {today} | {count} | ArXiv cs.CL/cs.CV/cs.HC + Semantic Scholar | "
            f"Automated weekly crawl — {count} new papers added |"
        )
        log_marker = "## Knowledge Update Log"
        table_header_pattern = r"\| Date \|.*\n\|[-|]+\|"
        if log_marker in content:
            log_idx = content.index(log_marker)
            header_match = re.search(table_header_pattern, content[log_idx:])
            if header_match:
                insert_at = log_idx + header_match.end() + 1
                content = content[:insert_at] + new_log_row + "\n" + content[insert_at:]
        self.brain_path.write_text(content, encoding="utf-8")
