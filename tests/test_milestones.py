"""Unit tests for craft.milestones."""

import pytest

from craft.milestones import M1, Milestone, Milestones, _has


# -------------------------------------------------------------------- _has


class TestHas:
    def test_finds_item_by_suffix(self):
        inv = {"minecraft:wooden_pickaxe": 1, "minecraft:dirt": 5}
        assert _has(inv, ":wooden_pickaxe")

    def test_missing_item(self):
        inv = {"minecraft:dirt": 5}
        assert not _has(inv, ":wooden_pickaxe")

    def test_empty_inv(self):
        assert not _has({}, ":wooden_pickaxe")

    def test_none_inv(self):
        assert not _has(None, ":wooden_pickaxe")

    def test_count_zero_ignored(self):
        inv = {"minecraft:wooden_pickaxe": 0, "minecraft:dirt": 5}
        assert not _has(inv, ":wooden_pickaxe")

    def test_suffix_distinguishes_iron_from_wooden(self):
        inv = {"minecraft:wooden_pickaxe": 1}
        assert _has(inv, ":wooden_pickaxe")
        assert not _has(inv, ":iron_pickaxe")
        assert _has(inv, "pickaxe")  # bare substring also matches via endswith


# ---------------------------------------------------------- M1 predicate


class TestM1Predicate:
    def test_needs_both_pickaxe_and_ticks(self):
        # No pickaxe → no fire
        assert not M1.predicate(
            {"inv": {"minecraft:dirt": 5}, "ticks_alive": 20000}
        )
        # No ticks → no fire
        assert not M1.predicate(
            {"inv": {"minecraft:wooden_pickaxe": 1}, "ticks_alive": 100}
        )
        # Both → fire
        assert M1.predicate(
            {"inv": {"minecraft:wooden_pickaxe": 1}, "ticks_alive": 12000}
        )

    def test_threshold_exact(self):
        state = {"inv": {"minecraft:wooden_pickaxe": 1}, "ticks_alive": 11999}
        assert not M1.predicate(state)
        state["ticks_alive"] = 12000
        assert M1.predicate(state)

    def test_iron_pickaxe_does_not_satisfy(self):
        """M1 specifically wants wooden_pickaxe — iron alone shouldn't satisfy."""
        # In practice an agent with iron has wooden too, but predicate semantics
        # should not silently widen.
        assert not M1.predicate(
            {"inv": {"minecraft:iron_pickaxe": 1}, "ticks_alive": 20000}
        )


# ----------------------------------------------------- Milestones.check


def _stats(day_count: int, day_ticks: int) -> dict:
    return {"day_count": day_count, "day_ticks": day_ticks}


class TestMilestonesCheck:
    def test_no_fire_at_spawn(self):
        ms = Milestones()
        inv = {"minecraft:wooden_pickaxe": 1}
        assert ms.check(_stats(0, 0), inv, turn=1) is None

    def test_fires_when_predicate_satisfied(self):
        ms = Milestones()
        # Spawn at day 0 ticks 0
        ms.check(_stats(0, 0), {"minecraft:wooden_pickaxe": 1}, turn=1)
        # Later: day 0 ticks 12000 = 12000 ticks_alive
        event = ms.check(
            _stats(0, 12000), {"minecraft:wooden_pickaxe": 1}, turn=20
        )
        assert event is not None
        assert event.name == "M1_iron_goal"
        assert event.turn == 20

    def test_fires_only_once(self):
        ms = Milestones()
        inv = {"minecraft:wooden_pickaxe": 1}
        ms.check(_stats(0, 0), inv, turn=1)  # anchor spawn
        ms.check(_stats(0, 12500), inv, turn=10)  # fires
        # Re-check — should not re-fire
        assert ms.check(_stats(0, 14000), inv, turn=11) is None
        assert "M1_iron_goal" in ms.fired

    def test_spawn_anchored_on_first_call(self):
        """Spawn anchor uses the first stats reading, not turn 0."""
        ms = Milestones()
        # First reading is day 100 ticks 0 — agent spawned mid-server-day
        ms.check(_stats(100, 0), {"minecraft:wooden_pickaxe": 1}, turn=1)
        # Same MC time relative to spawn = 12000 ticks; should fire
        event = ms.check(
            _stats(100, 12000), {"minecraft:wooden_pickaxe": 1}, turn=15
        )
        assert event is not None

    def test_night_spawn_doesnt_short_circuit(self):
        """Agent spawned at night shouldn't fire M1 just because day_ticks
        crosses 24000 immediately (would happen ~T1 on first day rollover)."""
        ms = Milestones()
        # Spawn at day 0 ticks 18000 (night)
        ms.check(_stats(0, 18000), {"minecraft:wooden_pickaxe": 1}, turn=1)
        # Crosses to day 1 ticks 3000 = only 9000 ticks alive → no fire
        assert ms.check(
            _stats(1, 3000), {"minecraft:wooden_pickaxe": 1}, turn=10
        ) is None
        # day 1 ticks 6000 = 12000 ticks alive → fires
        event = ms.check(
            _stats(1, 6000), {"minecraft:wooden_pickaxe": 1}, turn=14
        )
        assert event is not None

    def test_no_stats_returns_none(self):
        ms = Milestones()
        assert ms.check(None, {"minecraft:wooden_pickaxe": 1}, turn=1) is None

    def test_missing_day_fields_safe(self):
        """Partial stats (no day_count/day_ticks) shouldn't crash."""
        ms = Milestones()
        # Should not raise; just no fire because ticks_alive stays 0
        result = ms.check(
            {"health": 20.0}, {"minecraft:wooden_pickaxe": 1}, turn=1
        )
        assert result is None

    def test_fired_is_readonly_copy(self):
        """Mutating the returned fired dict shouldn't affect internal state."""
        ms = Milestones()
        ms.check(_stats(0, 0), {"minecraft:wooden_pickaxe": 1}, turn=1)
        ms.check(_stats(0, 12100), {"minecraft:wooden_pickaxe": 1}, turn=15)
        fired = ms.fired
        fired["fake"] = 999
        assert "fake" not in ms.fired


# ------------------------------------------- custom milestone chain


class TestCustomMilestones:
    def test_milestones_fire_in_order(self):
        """Multiple milestones: each fires at most once, earlier fires first."""
        fire_log = []

        m_a = Milestone(
            name="A",
            predicate=lambda s: s["ticks_alive"] >= 5000,
            message="A reached",
        )
        m_b = Milestone(
            name="B",
            predicate=lambda s: s["ticks_alive"] >= 10000,
            message="B reached",
        )
        ms = Milestones(milestones=[m_a, m_b])
        ms.check(_stats(0, 0), {}, turn=1)  # anchor
        e1 = ms.check(_stats(0, 6000), {}, turn=5)
        e2 = ms.check(_stats(0, 11000), {}, turn=10)
        assert e1 is not None and e1.name == "A"
        assert e2 is not None and e2.name == "B"
