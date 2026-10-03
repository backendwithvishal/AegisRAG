import time
import logfire


def parse_text(file_path: str) -> str:
    """
    Parses plain text and markdown (.txt, .md) files with UTF-8 encoding.
    Logs parse duration to Logfire.
    """
    start_time = time.time()
    with logfire.span("📄 Text Parsing", filename=file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            duration = time.time() - start_time
            logfire.info(f"✅ Text parsed {len(content)} characters in {duration:.3f}s from {file_path}")
            return content
        except Exception as e:
            duration = time.time() - start_time
            logfire.error(f"❌ Text Parse Failed for {file_path} after {duration:.3f}s: {e}")
            raise e
