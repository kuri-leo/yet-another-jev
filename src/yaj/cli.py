import click
import uvicorn

from yaj.client import LLMClient
from yaj.server import create_app
from yaj.strategy.logprob import LogprobStrategy
from yaj.strategy.structured import StructuredStrategy


@click.command()
@click.option(
    "--strategy",
    type=click.Choice(["logprob", "structured"]),
    required=True,
)
@click.option("--base-url", required=True, help="LLM API base URL")
@click.option("--api-key", required=True, help="LLM API key")
@click.option("--model", required=True, help="Model name")
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000, type=int)
def main(strategy, base_url, api_key, model, host, port):
    """Start the yet-another-jev decision server."""
    strat = LogprobStrategy() if strategy == "logprob" else StructuredStrategy()
    client = LLMClient(base_url=base_url, api_key=api_key, model=model)
    app = create_app(strat, client)
    uvicorn.run(app, host=host, port=port)
