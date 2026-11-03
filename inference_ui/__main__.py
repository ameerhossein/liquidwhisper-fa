"""Launch the inference server with the project's dotenv configuration."""
import argparse
import os

from src.dotenv import load_dotenv


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the voice inference UI")
    parser.add_argument("--host", default=os.environ.get("INFERENCE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("INFERENCE_PORT", "7861")))
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("inference_ui.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
