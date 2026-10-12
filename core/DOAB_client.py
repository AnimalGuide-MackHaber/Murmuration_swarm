import requests
import os
import time
from urllib.parse import quote
from typing import List, Optional
from core.settings_manager import SettingsManager
from core.swarm_logger import SwarmLogger

logger = SwarmLogger()

def fetch_doab_metadata(search_query: str) -> List[dict]:
    """
    Queries the DOAB REST API and returns the JSON payload containing bitstream data.
    """
    encoded_query = quote(search_query)
    url = f"https://directory.doabooks.org/rest/search?query={encoded_query}&expand=bitstreams"
    
    logger.log(f"Querying DOAB API: {url}", "INFO", "DOABClient")
    try:
        response = requests.get(url, headers={"Accept": "application/json"}, timeout=120)
        response.raise_for_status()
        data = response.json()
        logger.log(f"DOAB API Query Results: {data}", "DEBUG", "DOABClient")
        return data
    except Exception as e:
        logger.log(f"API Request Failed: {e}", "ERROR", "DOABClient")
        return []

def sanitize_filename(title: str) -> str:
    """
    Removes illegal filesystem characters from the DOAB item name.
    """
    keepcharacters = (' ', '.', '_', '-')
    sanitized = "".join(c for c in title if c.isalnum() or c in keepcharacters).rstrip()
    return sanitized if sanitized else "unknown_title"

def download_file(url: str, filepath: str, chunk_size: int = 8192) -> bool:
    """
    Downloads a file in chunks to prevent memory overload with large textbook PDFs.
    """
    try:
        with requests.get(url, stream=True, timeout=(10, 300)) as response:
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        logger.log(f"Download Error for {url}: {e}", "ERROR", "DOABClient")
        # Clean up partial files
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

class DOABClient:
    def __init__(self, download_dir: str = "swarm_data"):
        self.download_dir = download_dir
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
            
        settings = SettingsManager().get_settings()
        rl = settings.rate_limiting
        self.delay = rl.doab.request_delay_seconds if rl and hasattr(rl, 'doab') else 2.0
        
    def download_pdfs(self, search_query: str, limit: int = 3) -> List[str]:
        """
        Searches the DOAB API for books matching the search query,
        extracts the PDF bitstream, and downloads up to `limit` PDFs.
        Returns a list of local file paths.
        """
        os.makedirs(self.download_dir, exist_ok=True)
        downloaded_files = []
        
        logger.log(f"Starting download_pdfs for query: '{search_query}' (limit: {limit})", "INFO", "DOABClient")
        
        data = fetch_doab_metadata(search_query)
        if not data:
            logger.log(f"No metadata found for query: '{search_query}'", "WARNING", "DOABClient")
            return downloaded_files

        logger.log(f"Found {len(data)} items matching query. Processing up to {limit} PDFs...", "INFO", "DOABClient")

        for item_idx, item in enumerate(data):
            if len(downloaded_files) >= limit:
                logger.log(f"Limit reached ({limit}). Stopping.", "INFO", "DOABClient")
                break
                
            item_title = item.get('name', 'Unknown Title')
            bitstreams = item.get('bitstreams', [])
            
            logger.log(f"Processing item {item_idx+1}/{len(data)}: {item_title}", "DEBUG", "DOABClient")
            
            if not bitstreams:
                logger.log(f"No bitstreams for item: {item_title}", "DEBUG", "DOABClient")
                continue

            pdf_found = False
            for bitstream in bitstreams:
                if bitstream.get('mimeType') == 'application/pdf':
                    pdf_found = True
                    retrieve_path = bitstream.get('retrieveLink')
                    
                    if not retrieve_path:
                        logger.log(f"No retrieveLink for PDF in {item_title}", "WARNING", "DOABClient")
                        continue

                    full_url = f"https://directory.doabooks.org{retrieve_path}"
                    safe_title = sanitize_filename(item_title)
                    filepath = os.path.join(self.download_dir, f"{safe_title}.pdf")

                    if os.path.exists(filepath):
                        logger.log(f"File already exists: {filepath}", "INFO", "DOABClient")
                        downloaded_files.append(filepath)
                        break # Skip other bitstreams for this item
                    
                    logger.log(f"Downloading PDF from {full_url}", "INFO", "DOABClient")
                    success = download_file(full_url, filepath)
                    
                    if success:
                        logger.log(f"Successfully downloaded: {filepath}", "SUCCESS", "DOABClient")
                        downloaded_files.append(filepath)
                    else:
                        logger.log(f"Failed to download PDF for {item_title}", "ERROR", "DOABClient")
                    
                    # Rate limiting
                    if self.delay > 0:
                        logger.log(f"Waiting {self.delay}s (rate limit)...", "DEBUG", "DOABClient")
                        time.sleep(self.delay)
                    break # Only download first valid PDF bitstream per item

            if not pdf_found:
                 logger.log(f"No PDF bitstream found for item: {item_title}", "DEBUG", "DOABClient")

        logger.log(f"Operation complete. Total downloaded: {len(downloaded_files)}", "SUCCESS", "DOABClient")
        return downloaded_files

if __name__ == "__main__":
    # Example usage:
    # Target a specific topic. Using the title field reduces irrelevant metadata bleed.
    client = DOABClient(download_dir="./medical_pdfs")
    target_query = 'dc.title:"clinical anatomy"'
    client.download_pdfs(search_query=target_query, limit=3)