"""Health check script for the RAG Chatbot container."""
import sys
import urllib.request

def check_health() -> bool:
    """Check if Streamlit server is responding."""
    try:
        url = "http://localhost:8501/_stcore/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False

if __name__ == "__main__":
    if check_health():
        sys.exit(0)
    else:
        sys.exit(1)
