import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helpdesk.text import make_analyzer, light_stem

CFG = {"stemming": "light", "remove_stopwords": True}


def test_error_codes_stay_whole():
    assert make_analyzer(CFG)("Error 0x800CCC0E on VPN 809") == ["error", "0x800ccc0e", "vpn", "809"]


def test_stopwords_removed_and_case_folded():
    assert make_analyzer(CFG)("How do I reset THE password") == ["reset", "password"]


def test_light_stem_plurals_and_endings():
    assert light_stem("passwords") == "password"
    assert light_stem("devices") == "device"
    assert light_stem("access") == "access"        # -ss is not a plural
    assert light_stem("tickets") == "ticket"
    assert light_stem("logging") == "log"
    assert light_stem("0x80070005") == "0x80070005"


def test_stemming_can_be_switched_off():
    assert make_analyzer({"stemming": "none", "remove_stopwords": False})("the passwords") == ["the", "passwords"]
