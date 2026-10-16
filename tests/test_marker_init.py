from trf.encode import MARKER_TOKENS, marker_descriptions


def test_every_marker_has_a_description():
    d = marker_descriptions()
    assert set(d) == set(MARKER_TOKENS)
    assert all(v.strip() for v in d.values())


def test_case_tags_get_distinct_descriptions():
    d = marker_descriptions()
    case = [v for k, v in d.items() if k.startswith("<CASE-")]
    assert len(set(case)) == len(case)
