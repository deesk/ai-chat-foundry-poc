import asyncio
import os
import sys
from urllib.parse import urlparse
from dotenv import load_dotenv, find_dotenv
from azure.identity import AzureDeveloperCliCredential
from azure.ai.inference.aio import EmbeddingsClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.search_index_manager import SearchIndexManager
import json
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.azure', 'config.json')
with open(config_path) as f:
    env_name = json.load(f)['defaultEnvironment']
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.azure', env_name, '.env')
load_dotenv(env_path)

input_dir = "src/static/data"
output_file = "src/static/data/embeddings.csv"

async def main():
    tenant_id = os.getenv("AZURE_TENANT_ID")
    credential = AzureDeveloperCliCredential(tenant_id=tenant_id)
    project_endpoint = os.getenv("AZURE_EXISTING_AIPROJECT_ENDPOINT")
    inference_endpoint = f"https://{urlparse(project_endpoint).netloc}/models"

    embed = EmbeddingsClient(
        endpoint=inference_endpoint,
        credential=credential,
        credential_scopes=["https://ai.azure.com/.default"],
    )
    manager = SearchIndexManager(
        endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
        credential=credential,
        index_name=os.getenv("AZURE_AI_SEARCH_INDEX_NAME"),
        dimensions=int(os.getenv("AZURE_AI_EMBED_DIMENSIONS")),
        model=os.getenv("AZURE_AI_EMBED_DEPLOYMENT_NAME"),
        embeddings_client=embed
    )

    await manager.build_embeddings_file(
        input_directory=input_dir,
        output_file=output_file
    )
    await embed.close()
    print("Done — embeddings.csv updated")

if __name__ == "__main__":
    asyncio.run(main())