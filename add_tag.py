from huggingface_hub import HfApi
from dataclasses import dataclass
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--repo_id", type=str, required=True)
args = parser.parse_args()

@dataclass
class Args:
    repo_id: str
    """Path to the huggingface repo."""

hub_api = HfApi()

hub_api.create_tag(args.repo_id, tag="v3.0", repo_type="dataset")
refs = hub_api.list_repo_refs(args.repo_id, repo_type="dataset")

# tags are under refs.tags
print([tag.name for tag in refs.tags])
