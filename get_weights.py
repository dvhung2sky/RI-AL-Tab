"""Download the two TabICL v2 checkpoints the code loads from the repository root (each > 100 MB, so not committed).
python3 get_weights.py"""
import hashlib, os, shutil
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
# sha256 of the exact files used for the paper
FILES = {"tabicl-regressor-v2-20260212.ckpt": "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a",
         "tabicl-classifier-v2-20260212.ckpt": "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"}

for name, sha in FILES.items():
    dst = os.path.join(HERE, name)
    if not os.path.exists(dst):
        shutil.copy(hf_hub_download(repo_id="jingang/TabICL", filename=name), dst)
    h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    print(f"{name}: {'OK' if h == sha else 'SHA256 MISMATCH ' + h}")
