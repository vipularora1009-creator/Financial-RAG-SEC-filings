"""SEC Financial RAG: Question answering over SEC 10-K and 10-Q filings with citations."""

import sys


def main() -> None:
    """CLI entrypoint for sec-financial-rag."""
    import uvicorn

    port = 8000
    host = "0.0.0.0"
    for arg in sys.argv[1:]:
        if arg.startswith("--port="):
            port = int(arg.split("=")[1])
        elif arg.startswith("--host="):
            host = arg.split("=")[1]
        elif arg in ("-h", "--help"):
            print(
                "SEC Financial RAG: Question answering over SEC 10-K and 10-Q filings\n"
            )
            print("Usage:")
            print(
                "  sec-financial-rag [--host=0.0.0.0] [--port=8000]  Start the FastAPI server"
            )
            print(
                "  sec-financial-rag --help                          Show this help message"
            )
            return
    print(f"Starting SEC Financial RAG API server on http://{host}:{port} ...")
    uvicorn.run("sec_financial_rag.api.main:app", host=host, port=port, reload=False)
