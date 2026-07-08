"""
modal deployment for the cxr-vision inference api.

wraps api/main.py's fastapi app as a modal web function running on cpu only.

develop:
    modal serve modal_app/serve.py
deploy:
    modal deploy modal_app/serve.py
"""

import modal


def download_models():
    #baked into the image at build time so cold starts load local weights
    #instead of re-downloading ~700mb from openai/huggingface every time
    import clip
    from open_clip import create_model_from_pretrained

    clip.load("ViT-B/32", device="cpu")
    create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")


image = (
    modal.Image.debian_slim(python_version="3.11")
    #git is needed for the pip install below that clones openai/CLIP directly from github
    .apt_install("git")
    #cpu-only torch build, avoids pulling in cuda dependencies you don't need on modal
    .pip_install(
        "torch",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "fastapi[standard]",
        "python-multipart",
        "pillow",
        "numpy",
        "open_clip_torch",
        "transformers",
        "slowapi",
        "git+https://github.com/openai/CLIP.git",
    )
    .run_function(download_models)
    .add_local_python_source("src")
    .add_local_python_source("configs")
    #a flat module name here avoids needing an __init__.py under api/
    .add_local_file("api/main.py", remote_path="/root/api_main.py")
    .add_local_dir("outputs/checkpoints", remote_path="/root/outputs/checkpoints")
)

app = modal.App("cxr-vision-inference", image=image)


@app.function(cpu=2, memory=4096, scaledown_window=60, timeout=120)
@modal.asgi_app()
def fastapi_app():
    from api_main import app as web_app
    return web_app