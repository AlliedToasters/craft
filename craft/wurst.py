"""Pre-flight Wurst-hack toggling via homunculus's /wurst/* bridge.

Rollouts depend on Wurst modules being on (KillAura for mob auto-kill, AutoEat
for hunger management, AutoTool for combat/mining tool selection). These don't
auto-enable on a fresh MC session, so the harness asserts them on at startup
and records the result in the JSONL header.

The bridge is reflection-only on the homunculus side, so all errors here are
diagnostic — the harness should warn loudly when something's off but not block
the rollout unconditionally (a dev/test instance might not have Wurst at all).
"""

from __future__ import annotations

import requests

from craft.config import HOMUNCULUS_BASE

# Substrate-required: the agent's measured behavior depends on these being on.
# A rollout with any of them off is a different experiment, not a noisier one.
SURVIVAL_HACKS: tuple[str, ...] = (
    "KillAura",        # auto-attack hostile mobs in melee range
    "AutoEat",         # auto-consume edibles when hungry
    "AutoTool",        # auto-switch to the right tool for mining the current block
    "AutoSword",       # auto-switch to a sword when attacking — pairs with AutoTool/KillAura
    "AntiKnockback",   # suppress mob knockback (prevents fall-into-lava cascades)
    "AntiSpam",        # collapse spammy chat (keeps logs readable)
    "AutoDrop",        # tick-by-tick drop of items on its filter — seeded by autodrop policy
)
# AutoSwim was tried 2026-05-14 (after gemma R4 drowning) but removed
# 2026-05-16 — it thrashes against Baritone's swim/jump handling and
# produces a worse drowning outcome than no hack at all. Find a non-Wurst
# fix (Baritone water-traversal config, or LLM-level path planning).

# Observer convenience: doesn't change gameplay or substrate, just makes the
# rollout watchable. Kept separate so post-hoc analysis can distinguish
# "substrate gap" from "user couldn't see what happened".
OBSERVER_HACKS: tuple[str, ...] = (
    "Fullbright",      # brighten caves/night so the agent's actions are legible
)

# System-level automation glue: doesn't influence the agent's in-world behavior
# but keeps the rollout running through events that would otherwise require a
# human click. MC has no server-side force-respawn; AutoRespawn is the only
# way to get the player back into the world after a death without manual input.
SYSTEM_HACKS: tuple[str, ...] = (
    "AutoReconnect",   # auto-reconnect after server wipe / disconnect
    "AutoRespawn",     # click the respawn button automatically after death
)

REQUIRED_HACKS: tuple[str, ...] = SURVIVAL_HACKS + OBSERVER_HACKS + SYSTEM_HACKS


def set_hack(name: str, enabled: bool, *, timeout: float = 5.0) -> dict:
    """POST /wurst/hack. Returns the parsed response dict (always a dict).

    On transport error returns {"success": False, "reason": "transport_error",
    "message": str(e)} so callers can branch uniformly.
    """
    try:
        resp = requests.post(
            f"{HOMUNCULUS_BASE}/wurst/hack",
            json={"name": name, "enabled": enabled},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as e:
        return {"success": False, "reason": "transport_error", "message": str(e)}


def status(*, timeout: float = 5.0) -> dict:
    """GET /wurst/status. Returns parsed response or transport_error dict."""
    try:
        resp = requests.get(f"{HOMUNCULUS_BASE}/wurst/status", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as e:
        return {"success": False, "reason": "transport_error", "message": str(e)}


def set_item_list(
    hack: str,
    setting: str,
    items: list[str],
    *,
    op: str = "replace",
    timeout: float = 10.0,
) -> dict:
    """POST /wurst/setting for an ItemListSetting.

    Used by `seed_autodrop_from_tier()` to push the computed drop list into
    AutoDrop's `Items` setting. Returns the parsed response dict; transport
    errors fold into the standard {success: False, reason: "transport_error"}
    shape so callers can branch uniformly.
    """
    try:
        resp = requests.post(
            f"{HOMUNCULUS_BASE}/wurst/setting",
            json={"hack": hack, "setting": setting, "op": op, "value": items},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as e:
        return {"success": False, "reason": "transport_error", "message": str(e)}


def set_setting_value(
    hack: str,
    setting: str,
    value,
    *,
    timeout: float = 10.0,
) -> dict:
    """POST /wurst/setting for a non-list setting (Checkbox/Slider/Enum).

    Same endpoint as set_item_list but without an `op` field — homunculus
    stores the value verbatim, no merge semantics. Returns the parsed
    response dict; transport errors fold into the standard
    {success: False, reason: "transport_error"} shape so callers can branch
    uniformly.

    First user is craft.tools.handle_hunt_passive, which flips KillAura's
    "Filter passive mobs" checkbox false → attack → restore true.
    """
    try:
        resp = requests.post(
            f"{HOMUNCULUS_BASE}/wurst/setting",
            json={"hack": hack, "setting": setting, "value": value},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as e:
        return {"success": False, "reason": "transport_error", "message": str(e)}


def seed_autodrop_from_tier(tier: str, *, verbose: bool = True) -> dict:
    """Push the autodrop policy's drop-list-for-tier into AutoDrop.Items.

    Composes craft.autodrop's whitelist policy with the /wurst/setting
    transport. Returns {ok: bool, tier, drop_count, raw} so the caller can
    log + record in JSONL headers.

    AutoDrop must already be enabled (handled by ensure_hacks_on) — the
    setting is purely about policy, not module state.
    """
    from craft.autodrop import drop_list_for_tier  # avoid import cycle at module load

    drops = drop_list_for_tier(tier)
    resp = set_item_list("AutoDrop", "Items", drops, op="replace")
    ok = bool(resp.get("success"))
    if verbose:
        if ok:
            print(
                f"[autodrop] seeded tier={tier} drops={len(drops)} "
                f"(changed={resp.get('changed')})",
                flush=True,
            )
        else:
            print(
                f"[autodrop] FAILED tier={tier} reason={resp.get('reason')} "
                f"msg={resp.get('message', '')[:120]}",
                flush=True,
            )
    return {"ok": ok, "tier": tier, "drop_count": len(drops), "raw": resp}


def ensure_hacks_on(
    names: tuple[str, ...] = REQUIRED_HACKS,
    *,
    verbose: bool = True,
) -> dict:
    """Enable each hack in `names`, verify via /wurst/status, return a report.

    Report shape:
      {
        "ok": bool,                  # True iff every hack is on after toggling
        "wurst_loaded": bool,
        "results": [                 # per-hack outcome from /wurst/hack
          {"name": str, "enabled": bool, "changed": bool, "ok": bool, "raw": dict},
          ...
        ],
        "status_snapshot": dict,     # verbatim /wurst/status response (or error)
      }
    Never raises — errors land in the report. The caller decides whether to abort.
    """
    results: list[dict] = []
    wurst_loaded = True

    for name in names:
        r = set_hack(name, True)
        ok = bool(r.get("success") and r.get("enabled"))
        if r.get("reason") == "wurst_not_loaded":
            wurst_loaded = False
        entry = {
            "name": name,
            "enabled": bool(r.get("enabled")),
            "changed": bool(r.get("changed")),
            "ok": ok,
            "raw": r,
        }
        results.append(entry)
        if verbose:
            tag = "OK" if ok else "FAIL"
            extra = " (already on)" if ok and not entry["changed"] else ""
            reason = r.get("reason")
            if not ok and reason:
                extra = f" reason={reason} msg={r.get('message', '')[:80]}"
            print(f"[wurst] {tag} {name}{extra}", flush=True)

    snap = status() if wurst_loaded else {"success": False, "reason": "wurst_not_loaded"}
    all_ok = all(r["ok"] for r in results)
    if verbose and wurst_loaded and snap.get("success"):
        enabled_now = [h["name"] for h in snap.get("hacks", []) if h.get("enabled")]
        print(f"[wurst] {snap.get('enabled_count', '?')} hacks enabled: {enabled_now}",
              flush=True)
    elif verbose and not wurst_loaded:
        print("[wurst] !! Wurst not loaded at runtime — survival rollouts will run "
              "WITHOUT KillAura/AutoEat/AutoTool. Install the Wurst mod or expect "
              "very different outcomes.", flush=True)
    return {
        "ok": all_ok,
        "wurst_loaded": wurst_loaded,
        "results": results,
        "status_snapshot": snap,
    }


if __name__ == "__main__":
    # CLI for one-off probing: `python -m craft.wurst` prints status,
    # `python -m craft.wurst on KillAura` enables a hack.
    import json
    import sys

    if len(sys.argv) == 1:
        print(json.dumps(status(), indent=2))
    elif len(sys.argv) == 3 and sys.argv[1] in ("on", "off"):
        target = sys.argv[2]
        print(json.dumps(set_hack(target, sys.argv[1] == "on"), indent=2))
    elif len(sys.argv) == 2 and sys.argv[1] == "ensure":
        print(json.dumps(ensure_hacks_on(), indent=2))
    else:
        print(
            "usage:\n"
            "  python -m craft.wurst                # GET /wurst/status\n"
            "  python -m craft.wurst on  KillAura   # POST /wurst/hack enable\n"
            "  python -m craft.wurst off KillAura   # POST /wurst/hack disable\n"
            "  python -m craft.wurst ensure         # enable the REQUIRED_HACKS set",
            file=sys.stderr,
        )
        sys.exit(2)
