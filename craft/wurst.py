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

# Hacks that must be OFF every rollout. Wurst persists hack on/off state per
# profile, so a stale toggle survives client relaunches — and ensure_hacks_on
# only ever turns things ON. Sneak (perpetual crouch) got left enabled while
# debugging block placement; it caps movement at ~1.3 m/s and makes Baritone
# pathing crawl, silently degrading every agent. Force it off so a persisted
# toggle can't cripple the fleet again. (Scoped placement-sneak is handled
# correctly inside homunculus's Placer/BedPlacer; global Sneak is never wanted.)
FORBIDDEN_HACKS: tuple[str, ...] = ("Sneak",)


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


def set_autoeat_offhand_mode(*, verbose: bool = True) -> dict:
    """Restrict Wurst AutoEat to eat only from the hands/offhand.

    Sets "Take items from" = Hands (maxInvSlot=0, so AutoEat ignores all hotbar
    and main-inventory slots except the held one) and "Allow offhand" = true. The
    offhand then becomes the sole auto-eat source, which homunculus's offhand-food
    curator (Equipper) fills with policy-approved food. Net effect: raw meat sitting
    anywhere in the inventory is never auto-eaten — closing the "raw beef gets eaten
    before it can be cooked" hole. See FoodPolicy (homunculus) + set_food_policy.

    Returns {ok, take_items_from, allow_offhand}; never raises.
    """
    r1 = set_setting_value("AutoEat", "Take items from", "Hands")
    r2 = set_setting_value("AutoEat", "Allow offhand", True)
    ok = bool(r1.get("success")) and bool(r2.get("success"))
    if verbose:
        if ok:
            print("[autoeat] mode=Hands + offhand-eating on", flush=True)
        else:
            print(
                f"[autoeat] FAILED to set offhand mode: "
                f"take_items_from={r1.get('reason') or r1.get('success')} "
                f"allow_offhand={r2.get('reason') or r2.get('success')}",
                flush=True,
            )
    return {"ok": ok, "take_items_from": r1, "allow_offhand": r2}


def set_killaura_no_pvp(*, verbose: bool = True) -> dict:
    """Stop KillAura from attacking other players (enable "Filter players").

    Fleet agents share one MC server; with the player filter OFF (this build's
    default), KillAura melees any agent that wanders within range and
    AutoRespawn loops the kill — an inter-agent PvP confound that contaminates
    death tallies and is never wanted (they're all bots). Wurst filter semantics
    are EXCLUSION, so True = exclude players = no PvP. Harmless on a
    single-player instance (no other players to filter). Client-side only — no
    server.properties change.

    Returns the parsed /wurst/setting response; never raises.
    """
    r = set_setting_value("KillAura", "Filter players", True)
    if verbose:
        if r.get("success"):
            extra = "" if r.get("changed") else " (already)"
            print(f"[killaura] Filter players=on — no inter-agent PvP{extra}", flush=True)
        else:
            print(f"[killaura] FAILED to set Filter players: "
                  f"{r.get('reason')} {r.get('message', '')[:80]}", flush=True)
    return r


def set_killaura_attack_passives_default(*, verbose: bool = True) -> dict:
    """Set KillAura's ambient default to "attack passives" (Filter passive mobs = false).

    Pairs with ShearReflex (homunculus): when shears are in main hand, the reflex
    fires a shear interact on any sheep in 3.5m; with KillAura's passive filter OFF,
    KillAura ALSO attacks. Net effect: walkby-with-shears = wool drop + kill drop +
    meat — opportunistic harvest with no tool call required. Without shears, walkby
    is just "kill the sheep for meat", which is also the desired ambient.

    The explicit shear_sheep tool flips the filter back ON for the duration of its
    call (via _killaura_attack_passives) so the sheep survives — keeps the wool
    renewable when an agent deliberately commits to milking a sheep.

    Wurst filter semantics are EXCLUSION: True = exclude passives = don't attack.
    False = include = attack. So this sets "Filter passive mobs" = False.

    Returns the parsed /wurst/setting response; never raises.
    """
    r = set_setting_value("KillAura", "Filter passive mobs", False)
    if verbose:
        if r.get("success"):
            extra = "" if r.get("changed") else " (already)"
            print(
                f"[killaura] Filter passive mobs=off — ambient slaughter on{extra}",
                flush=True,
            )
        else:
            print(
                f"[killaura] FAILED to set Filter passive mobs: "
                f"{r.get('reason')} {r.get('message', '')[:80]}",
                flush=True,
            )
    return r


def set_food_policy(mode: str, *, verbose: bool = True, timeout: float = 5.0) -> dict:
    """POST /food_policy — homunculus endpoint (NOT /wurst/*), kept here so all
    preflight substrate config lives together.

    `mode` is "any" (daily-driver: every food approved for the offhand curator) or
    "cooked_only" (cook-capability tests: raw meat never staged → never auto-eaten).
    Pairs with set_autoeat_offhand_mode(). Returns the parsed response dict.
    """
    try:
        resp = requests.post(
            f"{HOMUNCULUS_BASE}/food_policy",
            json={"mode": mode},
            timeout=timeout,
        )
        resp.raise_for_status()
        r = resp.json()
    except (requests.RequestException, ValueError) as e:
        r = {"success": False, "reason": "transport_error", "message": str(e)}
    if verbose:
        if r.get("success"):
            print(f"[food_policy] mode={r.get('mode')}", flush=True)
        else:
            print(f"[food_policy] FAILED ({r.get('reason')}): {r.get('message', '')[:100]}",
                  flush=True)
    return r


def set_wurst_hud(visible: bool, *, verbose: bool = True, timeout: float = 5.0) -> dict:
    """POST /wurst/hud — show/hide Wurst's on-screen HUD (logo/hacklist/TabGui).

    The Wurst HUD is debug-only clutter on recorded rollouts, so preflight hides
    it (visible=False). homunculus already defaults it hidden, but setting it
    explicitly keeps the policy owned Python-side and deterministic regardless of
    the jar's default. Returns {success, visible, wurst_loaded} or a transport
    error dict; never raises.
    """
    try:
        resp = requests.post(
            f"{HOMUNCULUS_BASE}/wurst/hud",
            json={"visible": visible},
            timeout=timeout,
        )
        resp.raise_for_status()
        r = resp.json()
    except (requests.RequestException, ValueError) as e:
        r = {"success": False, "reason": "transport_error", "message": str(e)}
    if verbose:
        if r.get("success"):
            loaded = "" if r.get("wurst_loaded", True) else " (wurst not loaded — no visible effect)"
            print(f"[wurst_hud] visible={r.get('visible')}{loaded}", flush=True)
        else:
            print(f"[wurst_hud] FAILED ({r.get('reason')}): {r.get('message', '')[:100]}",
                  flush=True)
    return r


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


def ensure_hacks_off(
    names: tuple[str, ...] = FORBIDDEN_HACKS,
    *,
    verbose: bool = True,
) -> dict:
    """Disable each hack in `names`, return a report. Counterpart to
    ensure_hacks_on for hacks that must stay OFF.

    Wurst persists hack state across launches, so a stale toggle (e.g. Sneak
    left on during placement debugging) survives every relaunch and degrades
    the agent. This forces them off at preflight. Never raises.

    Report shape mirrors ensure_hacks_on (sans status_snapshot).
    """
    results: list[dict] = []
    wurst_loaded = True
    for name in names:
        r = set_hack(name, False)
        ok = bool(r.get("success")) and not r.get("enabled", False)
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
            extra = " (already off)" if ok and not entry["changed"] else ""
            reason = r.get("reason")
            if not ok and reason:
                extra = f" reason={reason} msg={r.get('message', '')[:80]}"
            print(f"[wurst] {tag} {name} forced-off{extra}", flush=True)
    return {
        "ok": all(r["ok"] for r in results),
        "wurst_loaded": wurst_loaded,
        "results": results,
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
