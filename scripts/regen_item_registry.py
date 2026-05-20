"""Regenerate craft/_item_registry.py from a Mojang server datagen --reports dump.

Used when the MC version bumps. End-to-end:

  1. On the MC server box, download the matching server.jar (URL from
     https://piston-meta.mojang.com/mc/game/version_manifest_v2.json
     under the target version's `downloads.server.url`):

       java -DbundlerMainClass=net.minecraft.data.Main \\
         -jar server.jar --reports --output /tmp/out

  2. Copy `/tmp/out/reports/registries.json` into this repo's `scratchpad/`.

  3. From the repo root, run:

       python scripts/regen_item_registry.py

Overwrites `craft/_item_registry.py` with the new frozen item list. Bumps
`MC_VERSION` if you pass `--version 1.X.Y` (otherwise the version string is
read from your previous registry file and printed for confirmation).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DEFAULT = REPO_ROOT / "scratchpad" / "registries.json"
DST = REPO_ROOT / "craft" / "_item_registry.py"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src", type=Path, default=SRC_DEFAULT,
        help="Path to registries.json (default: scratchpad/registries.json)",
    )
    ap.add_argument(
        "--version", default=None,
        help="MC version string to embed (default: re-read from existing registry)",
    )
    args = ap.parse_args()

    if not args.src.exists():
        print(f"[regen] source missing: {args.src}")
        print(f"[regen] dump it via: java -DbundlerMainClass=net.minecraft.data.Main "
              f"-jar server.jar --reports --output /tmp/out")
        return 1

    items = sorted(json.loads(args.src.read_text())["minecraft:item"]["entries"].keys())

    version = args.version
    if version is None and DST.exists():
        for line in DST.read_text().splitlines():
            if line.startswith("MC_VERSION"):
                version = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if version is None:
        version = "unknown"

    lines = [
        '"""Vanilla item registry, frozen from Mojang\'s server datagen --reports.',
        "",
        "Regenerate via scripts/regen_item_registry.py when the MC version bumps.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        f'MC_VERSION: str = "{version}"',
        "",
        "ALL_ITEMS: frozenset[str] = frozenset({",
    ]
    lines.extend(f'    "{it}",' for it in items)
    lines.append("})")
    lines.append("")
    DST.write_text("\n".join(lines))
    print(f"[regen] wrote {DST.relative_to(REPO_ROOT)} ({len(items)} items, version={version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
