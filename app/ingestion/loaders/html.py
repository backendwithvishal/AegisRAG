import time
from bs4 import BeautifulSoup 
import logfire


def parse_html(file_path: str) -> str:
    """
    Parses HTML content using BeautifulSoup.
    Cleans scripts, styles, meta, and noscript tags and extracts readable text for RAG.
    Logs parse duration to Logfire.
    """
    start_time = time.time()
    with logfire.span("📄 HTML Parsing", filename=file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, "html.parser")
            
            # 1. Remove Junk (Scripts, Styles, Metadata)
            for tag in soup(["script", "style", "meta", "noscript", "svg", "header", "footer", "nav"]):
                tag.decompose()
                
            # 2. Extract Text
            text = soup.get_text(separator="\n")
            
            # 3. Clean Whitespace (Collapse multiple newlines and extra spaces)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_clean = '\n'.join(chunk for chunk in chunks if chunk)
            
            duration = time.time() - start_time
            logfire.info(f"✅ HTML Parsed {len(text_clean)} chars from {file_path} in {duration:.3f}s")
            return text_clean
        except Exception as e:
            duration = time.time() - start_time
            logfire.error(f"❌ HTML Parse Failed for {file_path} after {duration:.3f}s: {e}")
            raise e
