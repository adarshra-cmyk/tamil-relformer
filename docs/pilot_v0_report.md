# Pilot v0: LLM-annotated round (provisional)

All labels here come from LLM annotation, not human annotators.
They are provisional and tagged `meta.annotator = "llm"` in
`data/processed/annotated_llm_v0.jsonl`. The blank sheets for human
annotators (`data/annotation/pilot_v0/pilot_sheet_{A,B}.csv`) are untouched.

## Process

1. 680 distant-supervision candidates (40 per relation, 120 NO_RELATION, at
   most one per article per label), DS label hidden from annotators.
2. Pass 1: six independent agents, about 114 items each, guide v0.
3. Pass 2: one more independent agent on the 136 double-annotation items.
4. Pass-1 batches applied span rules differently (dropped initials, bare
   place names before மாவட்டம்). Guide updated to v0.1 with explicit span and
   type rules; all 231 flagged or disagreeing items re-judged under v0.1.
   Attached initials (ஆர்.கே.நாராயணன்) still split the two reconciliation
   agents; decided as OK, same as separate initials.

## Numbers

| Measure | Value |
|---|---|
| Items usable (span and type OK) | 468 / 680 |
| Span or type errors | 212 / 680 (31%) |
| Pass 1 vs pass 2, labels on 83 items both labelled | raw 0.976, kappa 0.971 |
| Pass 1 vs pass 2, span/type flag | 129 / 136 agree |
| DS label equals annotator label | 266 / 468 (57%) |
| Stanza case correct on 936 entities | 0.919 |
| Stanza lemma correct on 936 entities | 0.872 |
| Sri Lankan Tamil items | 9 / 468 |

The pass 1 vs pass 2 kappa is model self-consistency, not inter-annotator
agreement; it is not a substitute for the human kappa in plan 5.3.

DS label correct by relation (pass 1, before reconciliation):
LOC:LOCATED_IN 38/40, LOC:COUNTRY 19/22, PER:EDUCATED_AT 27/34,
PER:EMPLOYER 23/30, PER:SPOUSE 20/32, PER:PARENT 20/32, PER:BIRTH_PLACE 19/34,
PER:NATIONALITY 16/28, ORG:FOUNDED_BY 14/30, PER:SIBLING 14/31,
PER:DEATH_PLACE 11/32, PER:RESIDENCE 8/24, ORG:HEADQUARTERS 7/15,
NO_RELATION 17/62. Most DS NO_RELATION items express a real relation that
Wikidata lacks (mostly LOC:LOCATED_IN).

## Usable label counts

NO_RELATION 150, LOC:LOCATED_IN 73, PER:EDUCATED_AT 30, PER:BIRTH_PLACE 29,
PER:EMPLOYER 28, LOC:COUNTRY 26, PER:SPOUSE 22, PER:NATIONALITY 21,
PER:PARENT 20, ORG:FOUNDED_BY 15, PER:SIBLING 14, PER:DEATH_PLACE 13,
PER:RESIDENCE 12, ORG:HEADQUARTERS 10, PER:MEMBER_OF 5.

## Candidate-generation problems found (fix before the next round)

1. Subject matched on surname only while the given name is written right
   before it (சஞ்சய் [E1: சிங்]).
2. Object matched inside a longer name (ரியல் மாட்ரிட், தில்லி உயர்நீதிமன்றம்,
   இந்திய தேசிய காங்கிரஸ்).
3. Common nouns in Wikidata aliases (கட்சி "party", degree words, role words).
4. Entity types taken from the relation slot are often wrong: parties and
   institutions typed LOCATION or PERSON (PER:MEMBER_OF: 35 of 40 unusable).

## Guide questions still open

* "Washington, D.C." and similar comma names.
* Same name for different places (சென்னை மாகாணம், Madras Presidency, vs the city).
* Father's name written in full instead of as an initial.
* Buildings vs institutions (cathedral, temple). Settled in guide v0.1: built places are LOCATION; entity_types.json types them via architectural structure / infrastructure.
* Whether "went to" a college states PER:EDUCATED_AT.
* Indian Administrative Service: employer or membership.
