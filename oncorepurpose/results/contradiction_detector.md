# Contradiction detector (full)

_2026-06-29T00:09:29.592614_

## What this is

OncoEvidence's *evidence-against* module. For known indications and for the current repurposing shortlist, it issues contradiction-oriented Europe PMC queries (resistance / ineffective / no benefit / did-not-improve / failed-trial) and grades each drug+cancer co-mention sentence as supporting / contradicting / neutral using lexical cues (no LLM). This makes the platform look for reasons a candidate might NOT work, not just reasons it might.

## Headline

- **True indications (60 pairs):** 12 have >=1 contradicting sentence; 19 contradicting vs 9 supporting sentences (mean contradiction fraction 0.169).
- **Shortlist candidates (30 pairs):** 2 have >=1 contradicting sentence; **0 are FLAGGED** for non-trivial contradicting evidence (>= 2 contradicting sentences and >= 34% of signed sentences contradicting).

## Flagged shortlist candidates

_No shortlist candidate crossed the flag threshold._

## Honest reading & caveats

- The classifier is **lexical**, not an LLM. 'Resistance' and 'refractory' appear heavily even for *effective* drugs (mechanism-of-resistance studies, second-line settings), so a non-zero contradiction tally on a true indication is expected and does NOT mean the drug fails. The signal is most useful as a relative flag (which candidates skew negative), not an absolute verdict.
- A negation guard drops cues like 'overcome resistance' / 'no resistance', but lexical matching still mislabels some sentences (irony, comparison to another arm, preclinical-vs-clinical).
- Only sentences co-mentioning the drug AND the cancer are counted, which is conservative (misses pronoun/abbreviation references) but reduces false hits.
- A flag is a prompt for human review, not evidence of inefficacy.
