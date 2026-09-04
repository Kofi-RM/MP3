"""Download only the required ViT files while building the deployment image."""
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
# Same model revision used and tested locally.
REVISION = '3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3'

if __name__ == '__main__':
    snapshot_download(
        repo_id='google/vit-base-patch16-224',
        revision=REVISION,
        local_dir=ROOT / '.models' / 'vit',
        allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors'],
    )
