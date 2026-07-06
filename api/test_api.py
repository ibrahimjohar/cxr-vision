"""
exercises all four api endpoints against a running local server and saves the
decoded images to disk for a visual sanity check.

usage:
    python test_api.py path/to/xray.png
"""

import os
import sys
import base64
import requests

BASE_URL = "http://127.0.0.1:8000"
OUT_DIR = "test_outputs"


def save_base64_png(b64_string, path):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_string))
    print(f"saved {path}")


def test_predict(image_path):
    print("\n[predict]")
    with open(image_path, "rb") as f:
        r = requests.post(f"{BASE_URL}/predict", files={"file": f})
    r.raise_for_status()
    data = r.json()
    print("classifier:", data["classifier"])
    print("vae reconstruction_mse:", data["vae"]["reconstruction_mse"])
    save_base64_png(data["unet"]["mask_png_base64"], f"{OUT_DIR}/unet_mask.png")
    save_base64_png(data["vae"]["reconstruction_png_base64"], f"{OUT_DIR}/vae_recon.png")


def test_clip(image_path):
    print("\n[clip]")
    with open(image_path, "rb") as f:
        r = requests.post(f"{BASE_URL}/clip", files={"file": f})
    r.raise_for_status()
    data = r.json()
    print("openai_clip:", data["openai_clip"])
    print("biomedclip:", data["biomedclip"])


def test_generate_gan():
    print("\n[generate/gan]")
    r = requests.post(f"{BASE_URL}/generate/gan")
    r.raise_for_status()
    save_base64_png(r.json()["image_png_base64"], f"{OUT_DIR}/gan_sample.png")


def test_generate_diffusion():
    print("\n[generate/diffusion]")
    r = requests.post(f"{BASE_URL}/generate/diffusion")
    r.raise_for_status()
    save_base64_png(r.json()["image_png_base64"], f"{OUT_DIR}/diffusion_sample.png")


def main():
    if len(sys.argv) < 2:
        print("usage: python test_api.py path/to/xray.png")
        sys.exit(1)

    image_path = sys.argv[1]
    os.makedirs(OUT_DIR, exist_ok=True)

    test_predict(image_path)
    test_clip(image_path)
    test_generate_gan()
    test_generate_diffusion()

    print(f"\ndone, check the {OUT_DIR} folder for saved images")


if __name__ == "__main__":
    main()