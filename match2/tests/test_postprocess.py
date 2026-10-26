"""後処理。値は試験のために作った物（主催の見本の答えは使わない）。"""
import pytest

from mc2ocr import postprocess as pp


@pytest.mark.parametrize("raw,want", [
    ("  8KQM417  ", "8KQM417"),
    ('"8KQM417"', "8KQM417"),
    ("`8KQM417`", "8KQM417"),
    ("```\n8KQM417\n```", "8KQM417"),
    ("Text: 8KQM417", "8KQM417"),
    ("Answer: 8KQM417.", "8KQM417"),
    ("TRANSCRIPTION: \"8KQM417\"", "8KQM417"),
    ("License plate: 8KQM417", "8KQM417"),
    ("The text is: 8KQM417", "8KQM417"),
    ("8KQM417。", "8KQM417"),
    ("YIELD\nTO\n  PEDESTRIANS", "YIELD TO PEDESTRIANS"),
    ("DETOUR   NEXT\r\nEXIT\n", "DETOUR NEXT EXIT"),
    ("", ""),
    (None, ""),
    ("苏 E·K3921", "苏 E·K3921"),
    ("「苏 E·K3921」", "苏 E·K3921"),
])
def test_clean_basics(raw, want):
    assert pp.clean(raw) == want


@pytest.mark.parametrize("raw,want", [
    ("CALIFORNIA 8KQM417", "8KQM417"),
    ("8KQM417 CALIFORNIA", "8KQM417"),
    ("NEW YORK QWT 4402", "QWT 4402"),
    ("New York QWT 4402", "QWT 4402"),
    ("WEST VIRGINIA 3PQX912", "3PQX912"),
    ("NORTH CAROLINA 3PQX912", "3PQX912"),
    ("TEXAS 6TNR205", "6TNR205"),
    ("OHIO 4821", "OHIO 4821"),          # 数字だけの語が 1 つ＝プレートの形ではない
    ("OREGON QWT 4402", "QWT 4402"),     # 英字の語＋数字の語の並び
    ("NEVADA 苏 E·K3921", "苏 E·K3921"),
])
def test_state_dropped_only_when_plate_like(raw, want):
    assert pp.clean(raw) == want


@pytest.mark.parametrize("raw", [
    "WRONG WAY",
    "ROAD CLOSED",
    "OHIO ROAD CLOSED",
    "TEXAS",
    "WASHINGTON",
    "NEW YORK",
    "DETOUR 2 MILES",
    "KEEP RIGHT 45",
    "MAINE 12",
])
def test_signs_and_lone_states_untouched(raw):
    assert pp.clean(raw) == raw


def test_state_table_has_50_states():
    assert len(pp.US_STATES) == 50 and len(set(pp.US_STATES)) == 50
    assert all(len(s.split()) == 2 for s in pp.US_STATES_2W) and all(len(s.split()) == 1 for s in pp.US_STATES_1W)


def test_every_state_is_dropped_next_to_a_plate_number():
    for st in pp.US_STATES:
        assert pp.clean(f"{st} 6TNR205") == "6TNR205", st


def test_state_never_dropped_when_it_would_leave_nothing():
    assert pp.clean("IDAHO") == "IDAHO"
