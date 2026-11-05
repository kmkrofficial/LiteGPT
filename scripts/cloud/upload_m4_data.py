"""Upload what M4 needs on the server to the private data repo, under m4/, with a checksum file, and prove every file by hash. Run on the laptop.

    python scripts/cloud/upload_m4_data.py --sft ~/m4_sft --items ~/m4_t2d/final

The M3 final checkpoint is already in the private runs repo, verified, so it is not uploaded again.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "cloud"))
REPO = "kmkrworks/litegpt-v2-data"


def main() -> int:
    from huggingface_hub import HfApi
    from m4_backup import hub_matches, sha256
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft", required=True)
    ap.add_argument("--items", required=True)
    a = ap.parse_args()
    files = {f"sft/{p.name}": p for p in sorted(Path(a.sft).iterdir()) if p.is_file()}
    files |= {f"t2d/{p.name}": p for p in sorted(Path(a.items).iterdir()) if p.is_file()}
    sums = Path(a.sft).parent / "m4_SHA256SUMS"
    sums.write_text("".join(f"{sha256(p)}  {dest}\n" for dest, p in files.items()), encoding="utf-8", newline="\n")
    api = HfApi()
    ok = True
    for dest, p in [*files.items(), ("SHA256SUMS", sums)]:
        path = f"m4/{dest}"
        info = api.get_paths_info(REPO, [path], repo_type="dataset")
        if not (info and hub_matches(info[0], p)):
            api.upload_file(path_or_fileobj=str(p), path_in_repo=path, repo_id=REPO, repo_type="dataset", commit_message=path)
            info = api.get_paths_info(REPO, [path], repo_type="dataset")
        good = bool(info and hub_matches(info[0], p))
        ok &= good
        print("verified" if good else "NOT VERIFIED", path, p.stat().st_size, flush=True)
    print("upload", "verified" if ok else "NOT VERIFIED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
