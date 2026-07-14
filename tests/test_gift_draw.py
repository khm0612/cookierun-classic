import numpy as np

from cookierun_bot.gift_draw import GiftDrawResult, draw_gifts, gift_button_visible


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def _frame(state: int):
    return np.full((100, 100, 3), state, np.uint8)


class GiftDevice:
    def __init__(self):
        self.state = 0
        self.taps = []

    def last_frame(self):
        return _frame(self.state)

    def tap(self, x, y):
        self.taps.append((self.state, x, y))
        if self.state == 0:          # menu gift button
            self.state = 1
        elif self.state == 1:        # Draw
            self.state = 2
        elif self.state == 2:        # gift box
            self.state = 3 if len([tap for tap in self.taps if tap[0] == 2]) == 1 else 4
        elif self.state == 3:        # Draw again
            self.state = 2
        elif self.state == 4:        # final Confirm
            self.state = 5


class GiftMatcher:
    def find(self, frame, name, threshold=0.8):
        state = int(frame[0, 0, 0])
        points = {
            (0, "giftbtn"): (30, 85),
            (0, "play"): (70, 70),
            (1, "draw"): (20, 20),
            (2, "giftbox"): (30, 30),
            (3, "drawagain"): (40, 40),
            (4, "confirm"): (50, 50),
        }
        return points.get((state, name))

    def present(self, frame, name, threshold=0.8):
        state = int(frame[0, 0, 0])
        return (state, name) in {(1, "giftdraw"), (2, "giftpick")}


def test_draw_gifts_spams_until_draw_again_is_gone():
    dev = GiftDevice()
    clock = FakeClock()

    result = draw_gifts(
        dev,
        GiftMatcher(),
        log=lambda _: None,
        sleep=clock.sleep,
        now=clock.now,
    )

    assert result == GiftDrawResult(draws=2, depleted=True, opened=True)
    assert dev.taps == [
        (0, 30, 89),
        (1, 20, 20),
        (2, 30, 30),
        (3, 40, 40),
        (2, 30, 30),
        (4, 50, 50),
    ]


class GiftPickOnlyMatcher:
    def find(self, frame, name, threshold=0.8):
        if int(frame[0, 0, 0]) == 0:
            if name == "giftbtn":
                return (30, 85)
            if name == "play":
                return (70, 70)
        return None

    def present(self, frame, name, threshold=0.8):
        return int(frame[0, 0, 0]) == 1 and name == "giftpick"


class GiftPickOnlyDevice:
    def __init__(self):
        self.state = 0
        self.taps = []

    def last_frame(self):
        return _frame(self.state)

    def tap(self, x, y):
        self.taps.append((self.state, x, y))
        self.state += 1


def test_draw_gifts_uses_box_fallback_only_on_verified_picker():
    dev = GiftPickOnlyDevice()
    clock = FakeClock()

    result = draw_gifts(
        dev,
        GiftPickOnlyMatcher(),
        log=lambda _: None,
        max_steps=4,
        sleep=clock.sleep,
        now=clock.now,
    )

    assert result.opened is True
    assert dev.taps[:2] == [(0, 30, 89), (1, 30, 50)]


def test_draw_gifts_does_not_claim_depleted_when_step_limit_expires():
    dev = GiftPickOnlyDevice()
    clock = FakeClock()

    result = draw_gifts(
        dev,
        GiftPickOnlyMatcher(),
        log=lambda _: None,
        max_steps=2,
        sleep=clock.sleep,
        now=clock.now,
    )

    assert result.draws == 1
    assert result.depleted is False


class RewardRevealDevice:
    def __init__(self):
        self.state = 0
        self.reward_reads = 0
        self.taps = []

    def last_frame(self):
        if self.state == 6:
            self.reward_reads += 1
            if self.reward_reads >= 4:
                self.state = 3
        return _frame(self.state)

    def tap(self, x, y):
        self.taps.append((self.state, x, y))
        if self.state == 0:
            self.state = 1
        elif self.state == 1:
            self.state = 2
        elif self.state == 2:
            self.state = 6       # reward reveal; boxes are still visible
        elif self.state == 3:
            self.state = 4
        elif self.state == 4:
            self.state = 5


class RewardRevealMatcher:
    def find(self, frame, name, threshold=0.8):
        state = int(frame[0, 0, 0])
        points = {
            (0, "giftbtn"): (30, 85),
            (0, "play"): (70, 70),
            (1, "draw"): (20, 20),
            (2, "giftbox"): (30, 30),
            (6, "giftbox"): (66, 66),
            (3, "drawagain"): (40, 40),
            (4, "confirm"): (50, 50),
        }
        return points.get((state, name))

    def present(self, frame, name, threshold=0.8):
        state = int(frame[0, 0, 0])
        return (state, name) in {(2, "giftpick"), (6, "giftpick")}


def test_draw_gifts_waits_out_reward_reveal_before_next_draw():
    dev = RewardRevealDevice()
    clock = FakeClock()

    result = draw_gifts(
        dev,
        RewardRevealMatcher(),
        log=lambda _: None,
        sleep=clock.sleep,
        now=clock.now,
    )

    assert result == GiftDrawResult(draws=1, depleted=True, opened=True)
    assert dev.taps == [
        (0, 30, 89),
        (1, 20, 20),
        (2, 30, 30),
        (3, 40, 40),
        (4, 50, 50),
    ]


def test_gift_button_visible_requires_template_match():
    assert gift_button_visible(_frame(0), GiftMatcher()) is True
    assert gift_button_visible(_frame(1), GiftMatcher()) is False


class WrongRegionGiftMatcher:
    def find(self, frame, name, threshold=0.8):
        return (90, 20) if name == "giftbtn" else None

    def present(self, frame, name, threshold=0.8):
        return False


def test_gift_button_visible_requires_bottom_bar_region():
    assert gift_button_visible(_frame(0), WrongRegionGiftMatcher()) is False


class RunningMatcher:
    def find(self, frame, name, threshold=0.8):
        return (10, 10) if name == "giftbtn" else None

    def present(self, frame, name, threshold=0.8):
        return name == "slide"


def test_draw_gifts_stops_if_run_starts():
    dev = GiftPickOnlyDevice()
    result = draw_gifts(dev, RunningMatcher(), log=lambda _: None)

    assert result == GiftDrawResult()
    assert dev.taps == []


class CardGameMatcher:
    def find(self, frame, name, threshold=0.8):
        if name == "giftbtn":
            return (30, 85)
        if name == "play":
            return None
        return None

    def present(self, frame, name, threshold=0.8):
        return name == "cardgame"


def test_draw_gifts_stops_on_card_bonus_without_tapping():
    dev = GiftPickOnlyDevice()
    result = draw_gifts(dev, CardGameMatcher(), log=lambda _: None)

    assert result == GiftDrawResult()
    assert dev.taps == []


class SelectModeMatcher:
    def find(self, frame, name, threshold=0.8):
        return (20, 20) if name in {"giftbtn", "selectmode"} else None

    def present(self, frame, name, threshold=0.8):
        return False


def test_draw_gifts_stops_on_select_mode_without_tapping():
    dev = GiftPickOnlyDevice()
    result = draw_gifts(dev, SelectModeMatcher(), log=lambda _: None)

    assert result == GiftDrawResult()
    assert dev.taps == []


class GiftCloseDevice:
    def __init__(self):
        self.state = 7
        self.taps = []

    def last_frame(self):
        return _frame(self.state)

    def tap(self, x, y):
        self.taps.append((self.state, x, y))
        if self.state == 7:
            self.state = 1
        elif self.state == 1:
            self.state = 2
        elif self.state == 2:
            self.state = 4
        elif self.state == 4:
            self.state = 5


class GiftCloseMatcher:
    def find(self, frame, name, threshold=0.8):
        state = int(frame[0, 0, 0])
        points = {
            (7, "giftclose"): (70, 70),
            (0, "play"): (70, 70),
            (1, "draw"): (20, 20),
            (2, "giftbox"): (30, 30),
            (4, "confirm"): (50, 50),
        }
        return points.get((state, name))

    def present(self, frame, name, threshold=0.8):
        return int(frame[0, 0, 0]) == 2 and name == "giftpick"


def test_draw_gifts_closes_pet_reward_modal_then_continues():
    dev = GiftCloseDevice()
    clock = FakeClock()

    result = draw_gifts(
        dev,
        GiftCloseMatcher(),
        log=lambda _: None,
        sleep=clock.sleep,
        now=clock.now,
    )

    assert result == GiftDrawResult(draws=1, depleted=True, opened=True)
    assert dev.taps == [
        (7, 70, 70),
        (1, 20, 20),
        (2, 30, 30),
        (4, 50, 50),
    ]


class DrawScreenWithCloseMatcher(GiftMatcher):
    def find(self, frame, name, threshold=0.8):
        state = int(frame[0, 0, 0])
        if state == 1 and name == "giftclose":
            return (70, 70)
        return super().find(frame, name, threshold)

    def present(self, frame, name, threshold=0.8):
        if int(frame[0, 0, 0]) == 1 and name == "giftdraw":
            return True
        return super().present(frame, name, threshold)


def test_draw_gifts_does_not_close_main_gift_draw_panel():
    dev = GiftDevice()
    clock = FakeClock()

    result = draw_gifts(
        dev,
        DrawScreenWithCloseMatcher(),
        log=lambda _: None,
        sleep=clock.sleep,
        now=clock.now,
    )

    assert result == GiftDrawResult(draws=2, depleted=True, opened=True)
    assert (1, 20, 20) in dev.taps
    assert (1, 70, 70) not in dev.taps


class ResultThenGiftDevice:
    def __init__(self):
        self.state = 8
        self.taps = []

    def last_frame(self):
        return _frame(self.state)

    def tap(self, x, y):
        self.taps.append((self.state, x, y))
        if self.state == 8:
            self.state = 0
        elif self.state == 0:
            self.state = 1
        elif self.state == 1:
            self.state = 2
        elif self.state == 2:
            self.state = 4
        elif self.state == 4:
            self.state = 5


class ResultThenGiftMatcher:
    def find(self, frame, name, threshold=0.8):
        state = int(frame[0, 0, 0])
        points = {
            (8, "ok"): (44, 44),
            (0, "giftbtn"): (30, 85),
            (0, "play"): (70, 70),
            (1, "draw"): (20, 20),
            (2, "giftbox"): (30, 30),
            (4, "confirm"): (50, 50),
        }
        return points.get((state, name))

    def present(self, frame, name, threshold=0.8):
        return int(frame[0, 0, 0]) == 2 and name == "giftpick"


def test_draw_gifts_can_clear_result_ok_before_opening_gifts():
    dev = ResultThenGiftDevice()
    clock = FakeClock()

    result = draw_gifts(
        dev,
        ResultThenGiftMatcher(),
        log=lambda _: None,
        sleep=clock.sleep,
        now=clock.now,
    )

    assert result == GiftDrawResult(draws=1, depleted=True, opened=True)
    assert dev.taps[:2] == [(8, 44, 44), (0, 30, 89)]
