import streamlit as st
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import time
from urllib.parse import urlparse

# --- Web Search Functionality ---

MAX_SEARCH_RESULTS = 3
MAX_CONTENT_LENGTH_PER_PAGE = 1500
REQUEST_TIMEOUT = 5

# Simple session cache (cleared on browser refresh/app restart)
search_cache = {}

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
    except ValueError:
        return False

def perform_web_search(query: str):
    """
    Performs a web search using DuckDuckGo, fetches content, and extracts text.
    Uses simple session caching.
    """
    global search_cache
    # Normalize query for caching
    cache_key = query.strip().lower()

    # Check cache first
    if cache_key in search_cache:
        print(f"Cache hit for web search: {query}")
        # Return a copy to prevent modification of cached data
        return [item.copy() for item in search_cache[cache_key]]

    print(f"Performing web search for: {query}")
    processed_results = []
    search_error = None

    try:
        # Use spinner for the search itself
        with st.spinner(f"Searching DuckDuckGo for '{query}'..."):
            with DDGS(timeout=10) as ddgs:
                ddgs_results = list(ddgs.text(query, max_results=MAX_SEARCH_RESULTS))

        if not ddgs_results:
            st.info(f"Web search for '{query}' returned no results.", icon="🦆")
            search_cache[cache_key] = [] # Cache empty result
            return []

        # Use spinner and progress for content fetching
        st.info(f"Found {len(ddgs_results)} results. Fetching content...", icon="🌐")
        fetch_progress = st.progress(0, text="Fetching content...") # Add text to progress

        with st.spinner("Fetching and parsing page content..."):
            for i, result in enumerate(ddgs_results):
                url = result.get('href')
                title = result.get('title', 'No Title')
                snippet = result.get('body', '')

                print(f"  Fetching content from: {url}")
                fetch_progress.progress((i + 0.5) / len(ddgs_results), text=f"Fetching: {url[:50]}...") # Update progress text
                extracted_text = f"Snippet: {snippet}" # Fallback

                if url and is_valid_url(url):
                    try:
                        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                        response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers, allow_redirects=True)
                        response.raise_for_status()
                        content_type = response.headers.get('content-type', '').lower()

                        if 'html' in content_type:
                            soup = BeautifulSoup(response.content, 'html.parser')
                            texts = []
                            for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'li', 'article', 'div']):
                                 if tag.name not in ['script', 'style', 'nav', 'footer'] and \
                                    not any(parent.name in ['script', 'style', 'nav', 'footer'] for parent in tag.parents):
                                     texts.append(tag.get_text(separator=' ', strip=True))
                            full_text = ' '.join(texts)
                            extracted_text = ' '.join(full_text.split())[:MAX_CONTENT_LENGTH_PER_PAGE]
                            if len(full_text) > MAX_CONTENT_LENGTH_PER_PAGE: extracted_text += "..."
                            print(f"    Successfully extracted ~{len(extracted_text)} chars from {url}")
                        else:
                            print(f"    Skipping non-HTML content ({content_type}) from {url}")
                            extracted_text = f"Content Type: {content_type}. Snippet: {snippet}"

                    except requests.exceptions.Timeout:
                        print(f"    Timeout fetching {url}")
                        extracted_text = f"Error: Timeout fetching content. Snippet: {snippet}"
                    except requests.exceptions.RequestException as e:
                        print(f"    Error fetching {url}: {e}")
                        extracted_text = f"Error fetching content ({e}). Snippet: {snippet}"
                    except Exception as e_parse:
                        print(f"    Error parsing {url}: {e_parse}")
                        extracted_text = f"Error parsing content. Snippet: {snippet}"
                else:
                     print(f"    Skipping invalid or missing URL for result: {title}")

                processed_results.append({
                    "title": title,
                    "url": url or "N/A",
                    "snippet": snippet,
                    "extracted_text": extracted_text
                })
                fetch_progress.progress((i + 1) / len(ddgs_results), text=f"Processed {i+1}/{len(ddgs_results)}")
                time.sleep(0.1)

        fetch_progress.empty() # Remove progress bar

    except Exception as e:
        search_error = f"An error occurred during web search: {e}"
        print(search_error)
        st.error(search_error, icon="🕸️")
        return None # Indicate total failure

    print(f"Web search processing complete. Returning {len(processed_results)} results.")
    # Store successful results in cache (make copies)
    search_cache[cache_key] = [item.copy() for item in processed_results]
    return processed_results


def format_search_results(results: list[dict]):
    """Formats web search results into a string for LLM context."""
    if not results:
        return ""
    context_str = "\n\n---\nWeb Search Results (Top {}):\n---\n".format(len(results))
    for i, result in enumerate(results):
        context_str += f"[{i+1}] Title: {result.get('title', 'N/A')}\n"
        context_str += f"    URL: {result.get('url', 'N/A')}\n"
        content_preview = result.get('extracted_text', 'No content extracted.')
        context_str += f"    Content Preview: {content_preview}\n\n"
    context_str += "---\nEnd of Web Search Results\n---\n"
    return context_str.strip()