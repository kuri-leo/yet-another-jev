import logging

import click
import uvicorn
from dotenv import load_dotenv

from yaj.client import LLMClient
from yaj.server import create_app
from yaj.strategy.logprob import LogprobStrategy
from yaj.strategy.structured import StructuredStrategy

# Load .env before Click reads envvar defaults
load_dotenv()


@click.command()
@click.option(
    "--strategy",
    type=click.Choice(["logprob", "structured"]),
    required=True,
)
@click.option(
    "--base-url",
    envvar="YAJ_BASE_URL",
    required=True,
    help="LLM API base URL (env: YAJ_BASE_URL)",
)
@click.option(
    "--api-key",
    envvar="YAJ_API_KEY",
    required=True,
    help="LLM API key (env: YAJ_API_KEY)",
)
@click.option(
    "--model",
    envvar="YAJ_MODEL",
    required=True,
    help="Model name (env: YAJ_MODEL)",
)
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000, type=int)
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(strategy, base_url, api_key, model, host, port, debug):
    """Start the yet-another-jev decision server."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )
    strat = LogprobStrategy() if strategy == "logprob" else StructuredStrategy()
    client = LLMClient(base_url=base_url, api_key=api_key, model=model)
    app = create_app(strat, client)
    uvicorn.run(app, host=host, port=port)
