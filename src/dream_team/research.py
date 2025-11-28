"""
Research module for Dream Team framework.

Integrates with Semantic Scholar API and uses LLM to analyze papers.
Features robust backoff, venue tracking, citation sorting, and context formatting.
"""

import requests
import time
import backoff  # Requires: pip install backoff
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def on_backoff(details: Dict) -> None:
    """Callback to log backoff events when API requests fail."""
    print(
        f"⚠️  Request failed. Backing off {details['wait']:0.1f}s after {details['tries']} tries "
        f"calling {details['target'].__name__}..."
    )

# -------------------------------------------------------------------------
# Data Structures
# -------------------------------------------------------------------------

@dataclass
class PaperResult:
    """Semantic Scholar paper result data structure."""
    paper_id: str
    title: str
    authors: List[str]
    year: int
    venue: str  # Feature 1: Added Venue field
    abstract: str
    citation_count: int
    influential_citation_count: int
    url: str

    def to_paper(self) -> 'Paper':
        """Convert to internal Paper object format."""
        # Lazy import to avoid circular dependency issues
        from .agent import Paper
        return Paper(
            title=self.title,
            authors=self.authors,
            year=self.year,
            abstract=self.abstract,
            semantic_scholar_id=self.paper_id,
            citation_count=self.citation_count,
            relevance_score=0.0  # To be computed by LLM
        )

# -------------------------------------------------------------------------
# API Wrapper
# -------------------------------------------------------------------------

class SemanticScholarAPI:
    """Wrapper for Semantic Scholar API with robust retries and sorting."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"x-api-key": self.api_key})
        
        # Polite throttling (still useful as a baseline even with backoff)
        self.rate_limit_delay = 1.0

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, requests.exceptions.Timeout),
        max_tries=5,
        on_backoff=on_backoff
    )
    # Feature 4: Robust Decorator-based Backoff
    def search(
        self,
        query: str,
        limit: int = 10,
        year_range: Optional[tuple] = None,
        fields: List[str] = None
    ) -> List[PaperResult]:
        """
        Search for papers.
        Automatically sorts results by citation count to prioritize impact.
        """

        # Feature 1: Added 'venue' to default fields
        if fields is None:
            fields = ["paperId", "title", "authors", "venue", "year", "abstract", 
                      "citationCount", "influentialCitationCount", "url"]

        params = {
            "query": query,
            "limit": limit,
            "fields": ",".join(fields)
        }

        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"

        # Polite delay before request to respect standard API norms
        time.sleep(self.rate_limit_delay)

        response = self.session.get(
            f"{self.BASE_URL}/paper/search",
            params=params,
            timeout=10
        )
        response.raise_for_status()

        data = response.json()
        results = []

        for paper_data in data.get("data", []):
            if not paper_data.get("abstract"):
                continue  # Skip papers without abstracts

            results.append(PaperResult(
                paper_id=paper_data["paperId"],
                title=paper_data["title"],
                authors=[a.get("name", "Unknown") for a in paper_data.get("authors", [])],
                year=paper_data.get("year", 0),
                venue=paper_data.get("venue", "Unknown Venue"), # Capture Venue
                abstract=paper_data.get("abstract", ""),
                citation_count=paper_data.get("citationCount", 0),
                influential_citation_count=paper_data.get("influentialCitationCount", 0),
                url=paper_data.get("url", "")
            ))

        # Feature 2: Sort by Citation Count (Descending)
        # This ensures we process the most impactful papers first
        results.sort(key=lambda x: x.citation_count or 0, reverse=True)

        return results

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.HTTPError, requests.exceptions.ConnectionError),
        max_tries=3,
        on_backoff=on_backoff
    )
    def get_paper(self, paper_id: str) -> Optional[PaperResult]:
        """Get specific paper by ID."""
        fields = ["paperId", "title", "authors", "venue", "year", "abstract", 
                  "citationCount", "influentialCitationCount", "url"]

        time.sleep(self.rate_limit_delay)
        response = self.session.get(
            f"{self.BASE_URL}/paper/{paper_id}",
            params={"fields": ",".join(fields)},
            timeout=10
        )
        response.raise_for_status()
        
        paper_data = response.json()
        return PaperResult(
            paper_id=paper_data["paperId"],
            title=paper_data["title"],
            authors=[a.get("name", "Unknown") for a in paper_data.get("authors", [])],
            year=paper_data.get("year", 0),
            venue=paper_data.get("venue", "Unknown Venue"),
            abstract=paper_data.get("abstract", ""),
            citation_count=paper_data.get("citationCount", 0),
            influential_citation_count=paper_data.get("influentialCitationCount", 0),
            url=paper_data.get("url", "")
        )

# -------------------------------------------------------------------------
# Research Assistant Agent
# -------------------------------------------------------------------------

class ResearchAssistant:
    """High-level research operations using Semantic Scholar + LLM."""

    def __init__(self, ss_api: SemanticScholarAPI, llm: 'GeminiLLM'):
        self.ss_api = ss_api
        self.llm = llm

    def format_results_context(self, results: List[PaperResult]) -> str:
        """
        Feature 5: Create a single string summary of papers.
        Useful for feeding a condensed list of papers to the LLM context.
        """
        paper_strings = []
        for i, paper in enumerate(results):
            # Limit to 3 authors for brevity in context window
            authors_str = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors_str += " et al."
                
            entry = (
                f"{i + 1}: {paper.title}\n"
                f"   Authors: {authors_str}\n"
                f"   Venue: {paper.venue}, {paper.year}\n"
                f"   Citations: {paper.citation_count}\n"
                f"   Abstract: {paper.abstract[:200]}..." # Truncate abstract for overview
            )
            paper_strings.append(entry)
        
        return "\n\n".join(paper_strings)

    def research_topic(
        self,
        query: str,
        context: str = "",
        num_papers: int = 5,
        year_range: tuple = (2020, 2025)
    ) -> List['Paper']:
        """
        Research a topic: search papers + extract key insights using LLM.
        """
        from .agent import Paper  # Lazy import

        print(f"🔍 Researching: '{query}'")

        # 1. Search (Now robust & sorted by popularity)
        # Fetch 2x the requested amount to allow for filtering
        results = self.ss_api.search(
            query=query,
            limit=num_papers * 2,
            year_range=year_range
        )

        if not results:
            print("   No papers found")
            return []

        print(f"   Found {len(results)} papers (sorted by impact).")

        # Feature 5 Usage: Create a context block for the LLM if needed for meta-analysis
        # (This variable 'papers_context' can be used if you change the prompt to select papers first)
        papers_context = self.format_results_context(results[:num_papers])

        # Use LLM to analyze relevance and extract insights
        papers = []
        
        # Iterate through the top N papers (which are now sorted by citation count)
        for i, result in enumerate(results[:num_papers]):
            print(f"   Analyzing paper {i+1}/{min(num_papers, len(results))}: {result.title[:60]}...")

            # Enhanced prompt with Venue and Citations
            analysis_prompt = f"""You are a research assistant analyzing a scientific paper for relevance and insights.

Context: {context}

Paper Title: {result.title}
Venue/Journal: {result.venue}
Year: {result.year}
Citations: {result.citation_count}
Authors: {', '.join(result.authors)}
Abstract: {result.abstract}

Tasks:
1. Rate relevance to the context (0.0-1.0). Consider the Venue quality and Citation count in your assessment of reliability.
2. Extract 3-5 key findings that are actionable.
3. List specific techniques/methods mentioned.
4. Provide a brief summary of applicability.

Respond in JSON format:
{{
    "relevance_score": 0.0-1.0,
    "key_findings": ["finding 1", "finding 2", ...],
    "techniques": ["technique 1", "technique 2", ...],
    "applicability": "brief summary"
}}
"""

            try:
                # Assuming llm.generate_json exists in your LLM wrapper
                analysis = self.llm.generate_json(analysis_prompt, temperature=0.3)

                paper = Paper(
                    title=result.title,
                    authors=result.authors,
                    year=result.year,
                    abstract=result.abstract,
                    key_findings=analysis.get("key_findings", []),
                    techniques=analysis.get("techniques", []),
                    relevance_score=analysis.get("relevance_score", 0.0),
                    semantic_scholar_id=result.paper_id,
                    citation_count=result.citation_count
                )
                
                # Store the venue in the Paper object if the Paper class supports it,
                # otherwise it's just available in the PaperResult.
                # paper.venue = result.venue 

                papers.append(paper)

            except Exception as e:
                print(f"   ⚠️  Error analyzing paper: {e}")
                # Fallback: create paper without LLM analysis
                paper = Paper(
                    title=result.title,
                    authors=result.authors,
                    year=result.year,
                    abstract=result.abstract,
                    semantic_scholar_id=result.paper_id,
                    citation_count=result.citation_count,
                    relevance_score=0.5
                )
                papers.append(paper)

        # Re-sort by relevance score (LLM's judgment) instead of raw citation count
        papers.sort(key=lambda p: p.relevance_score, reverse=True)

        avg_score = sum(p.relevance_score for p in papers) / len(papers) if papers else 0
        print(f"   ✅ Analyzed {len(papers)} papers (avg relevance: {avg_score:.2f})")

        return papers

# -------------------------------------------------------------------------
# Singleton Pattern
# -------------------------------------------------------------------------

_research_assistant = None

def get_research_assistant() -> ResearchAssistant:
    """Get or create global research assistant."""
    global _research_assistant
    if _research_assistant is None:
        from .llm import get_llm
        
        # Use API key from environment if available
        api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
        ss_api = SemanticScholarAPI(api_key=api_key)
        llm = get_llm()
        
        _research_assistant = ResearchAssistant(ss_api, llm)
    return _research_assistant