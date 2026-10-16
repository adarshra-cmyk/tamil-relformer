# TamilRelFormer annotation guide, v0.1 (pilot)

Status: draft for the Month 2 pilot. Definitions will change after the pilot
agreement check. Relation inventory: `configs/relations.json`.

## 1. The task

Each item shows one Tamil sentence with two marked entities:

```
[E1: சாம்] [E2: யாழ்ப்பாணத்தில்] வாழ்ந்தார்.
```

Pick the relation that **this sentence states** between E1 (subject) and E2
(object). Use only what the sentence says. If you know from outside that Sam
lives in Jaffna but the sentence does not say it, the answer is NO_RELATION.

Direction matters. The relation is always read as "E1 RELATION E2".
`PER:PARENT` means E2 is a parent of E1.

## 2. Labels

| Label | E1 / E2 types | Use when the sentence says... |
|---|---|---|
| PER:BIRTH_PLACE | PERSON / LOCATION | E1 was born in E2 |
| PER:DEATH_PLACE | PERSON / LOCATION | E1 died in E2 |
| PER:RESIDENCE | PERSON / LOCATION | E1 lives or lived in E2 (not just visited, worked or spoke there) |
| PER:NATIONALITY | PERSON / LOCATION | E1 is a citizen or national of country E2 |
| PER:EDUCATED_AT | PERSON / ORGANIZATION | E1 studied at E2 |
| PER:EMPLOYER | PERSON / ORGANIZATION | E1 works or worked for E2 |
| PER:MEMBER_OF | PERSON / ORGANIZATION | E1 is a member of E2 (party, team, association) |
| PER:SPOUSE | PERSON / PERSON | E1 and E2 are or were married |
| PER:PARENT | PERSON / PERSON | E2 is the father or mother of E1 |
| PER:SIBLING | PERSON / PERSON | E1 and E2 are siblings |
| ORG:HEADQUARTERS | ORGANIZATION / LOCATION | E1 is based or headquartered in E2 |
| ORG:FOUNDED_BY | ORGANIZATION / PERSON | E2 founded E1 |
| LOC:LOCATED_IN | LOCATION / LOCATION | E1 is inside E2 (district, state, province) |
| LOC:COUNTRY | LOCATION / LOCATION | E1 is in country E2 |
| NO_RELATION | any | none of the above is stated |

If two labels fit (for example a person born in and living in the same town,
and the sentence says both), choose the more specific one stated by the main
verb and write the other in `notes`.

## 3. Case suffixes are evidence, not the answer

Tamil case endings help but do not decide the label:

| Sentence | Label |
|---|---|
| சாம் யாழ்ப்பாணத்தில் வாழ்ந்தார். (lived in Jaffna) | PER:RESIDENCE |
| சாம் யாழ்ப்பாணத்தில் பேசினார். (spoke in Jaffna) | NO_RELATION |
| சாம் யாழ்ப்பாணத்தில் பிறந்தார். (was born in Jaffna) | PER:BIRTH_PLACE |
| சாம் யாழ்ப்பாணத்திலிருந்து வந்தார். (came from Jaffna) | NO_RELATION unless the sentence says he lived or was born there |

All four have Jaffna with a locative or ablative ending. The verb decides.

### Nationality adjectives

A country adjective that describes the person directly states
PER:NATIONALITY: [E1: X] ஒரு [E2: இந்திய] நடிகை ("an Indian actress"),
[E2: அமெரிக்க] எழுத்தாளர் [E1: Y]. When the adjective belongs to something
else in the phrase, it does not: [E2: இந்தியத்] திரைப்படப் பின்னணிப் பாடகர்
("playback singer in Indian films"), [E2: இந்திய] விடுதலைப் போராட்ட வீராங்கனை
("fighter in the Indian freedom struggle"), [E2: இலங்கை] கொழும்புப் பிரதேச
அணியின் துடுப்பாட்டக்காரர் ("player for a Sri Lankan regional team"). These
only suggest nationality (foreign nationals worked in Indian films and the
freedom struggle, and play for domestic teams), so they are NO_RELATION.

## 4. Entity spans

* A span is the whole word as written, including its case suffix:
  `யாழ்ப்பாணத்தில்`, not `யாழ்ப்பாணம்`.
* Multi-word names include all words: `சுப்பிரமணிய பாரதியின்`.
* If a span is wrong (cuts a name, includes extra words, or marks the wrong
  thing), set `span_ok = no` and write the correct span in `notes`. Do not
  label the relation for that item.
* If an entity type is wrong (a place marked as PERSON), set `type_ok = no`.

Span rules added in v0.1 after the first pilot pass showed annotators
deciding these cases differently:

| Case | Example | span_ok |
|---|---|---|
| separate initials left out | சு. [E1: சமுத்திரம்] | yes |
| surname marked, given name written right before it | சஞ்சய் [E1: சிங்] | no |
| bare place name before a type word | [E2: கரூர்] மாவட்டம் | yes |
| part of a longer name | ரியல் [E2: மாட்ரிட்] (Real Madrid), [E2: தில்லி] உயர்நீதிமன்றம் | no |
| a common noun marked as an entity | [E2: கட்சி] ("party") | no |

Types: political parties, schools, colleges, universities, companies, teams
and religious bodies are ORGANIZATION. Buildings and other built places
(temples, churches, dams, airports, bridges, stadiums) are LOCATION, unless the
sentence is about the body that runs them (a temple trust appointing a head) -
then ORGANIZATION. Gods and mythological or legendary figures (Rama, Arjuna,
Zeus) are PERSON. Festivals, languages, religions as belief systems, ideas and animals are none of the
types (type_ok = no).

## 5. Morphology check (optional columns)

For each entity, the sheet shows the analyzer's lemma and case
(`pred_subj_lemma`, `pred_subj_case`, ...). If you have time, fill
`subj_lemma`, `subj_case`, `obj_lemma`, `obj_case` with the correct values.
Leave them blank if the prediction is right. These give us "verified"
morphology to compare with analyzer output (plan section 7).

Case values: Nom, Acc, Dat, Loc, Ins, Com, Abl, Gen, Ben, Voc, OTHER.

## 6. Variety

Fill `variety` with `Sri Lankan Tamil`, `Indian Tamil` or `unknown`, based on
the language of the sentence, not the topic. A sentence about Colombo written
in standard Indian Tamil is `Indian Tamil` or `unknown`. If unsure, use
`unknown`.

## 7. Quality process

* 15 to 20 percent of items are labelled by two annotators independently.
* Disagreements go to adjudication; the adjudicated label is final.
* Target Cohen's kappa of about 0.70 or higher. If a relation pair is often
  confused, the definitions get revised before the main round.
* Do not discuss specific items with the other annotator before both have
  finished.

## 8. Columns in the sheet

| Column | Fill? | Meaning |
|---|---|---|
| item_id | no | item identifier |
| sentence_marked | no | sentence with [E1: ...] and [E2: ...] |
| subj_type, obj_type | no | entity types |
| label | **yes** | one label from section 2 |
| span_ok, type_ok | **yes** | yes / no |
| variety | **yes** | section 6 |
| pred_* | no | analyzer morphology |
| subj_lemma ... obj_case | optional | section 5 |
| notes | optional | anything unclear |
