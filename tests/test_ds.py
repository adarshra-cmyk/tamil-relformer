import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_ds_candidates import WORD_RE, Gazetteer, split_sentences


def words(s):
    return [(m.group(), m.start(), m.end()) for m in WORD_RE.finditer(s)]


def test_inflected_matches():
    g = Gazetteer()
    for name, q in [("யாழ்ப்பாணம்", "J"), ("கொழும்பு", "C"), ("குமார்", "K"), ("சென்னை", "M"),
                    ("சுப்பிரமணிய பாரதி", "B"), ("மட்டக்களப்பு", "B2")]:
        g.add(name, q)
    cases = {
        "சாம் யாழ்ப்பாணத்தில் வாழ்ந்தார்.": ("J", "யாழ்ப்பாணத்தில்"),
        "அவன் கொழும்பிலிருந்து வந்தான்.": ("C", "கொழும்பிலிருந்து"),
        "குமாருக்கு கொடுத்தேன்.": ("K", "குமாருக்கு"),
        "அவர் சென்னையில் பிறந்தார்.": ("M", "சென்னையில்"),
        "சுப்பிரமணிய பாரதியின் கவிதைகள்.": ("B", "சுப்பிரமணிய பாரதியின்"),
        "அவள் மட்டக்களப்பிலிருந்து வந்தாள்.": ("B2", "மட்டக்களப்பிலிருந்து"),
    }
    for sent, (q, span) in cases.items():
        hits = g.find(words(sent))
        assert [(sent[s:e], k) for s, e, k, _ in hits] == [(span, q)], sent


def test_no_false_compound_match():
    g = Gazetteer(); g.add("ராமன்", "R")
    assert g.find(words("ராமநாதபுரம் ஒரு நகரம்.")) == []


def test_initials_not_split():
    s = split_sentences("வ. உ. சிதம்பரம் பிள்ளை ஒரு தலைவர். அவர் தூத்துக்குடியில் வாழ்ந்தார்.")
    assert len(s) == 2 and s[0].startswith("வ. உ.")


def test_rejects_shared_prefix_words():
    g = Gazetteer(); g.add("பெரு", "PERU"); g.add("பிரான்சு", "FR"); g.add("இந்தியா", "IN")
    assert g.find(words("அவரது பெரும்பாலான படைப்புகள்.")) == []
    assert g.find(words("போனி பிரான்செஸ்கா ரைட் ஒரு நடிகை.")) == []
    hits = g.find(words("அவர் இந்தியாவைச் சேர்ந்தவர். பிரான்சிலிருந்து வந்தார்."))
    assert [k for *_, k, _ in hits] == ["IN", "FR"]


def test_multiword_needs_whitespace_gap():
    g = Gazetteer(); g.add("ஆலங்குடி சிவகங்கை", "A")
    s = "ஆலங்குடி (சிவகங்கை) ஒரு ஊர்."
    assert g.find(words(s), s) == []
