# Screening log

Run 2026-09-30. Protocol: AUDIT_PROTOCOL.md (SHA-256 32b85ead...a7f7, matches AUDIT_SEAL.sha256; not edited).
Screening only (inclusion and full-text access), no coding of A1-A11. Full records: screening.json, in screening order.

## Method notes
- Shell egress to publishers was blocked, so all checks used the web search and page-fetch tools. A paper was included
  only if the fetched full text showed methods and experimental results.
- Duplicates (same work already screened, including PMC/PubMed/arXiv/ResearchGate copies) were skipped and are not
  listed in screening.json.
- No candidate had Khezri or Trik as an author. (Full author list not captured for C11; the visible authors are not them.)
- Some pages could not be read: Springer, doi.org and ResearchGate returned HTTP 429 (rate limit) during the session,
  PMC returned reCAPTCHA, and IEEE Xplore returned error 418. Where no other copy could be opened, the candidate is
  excluded under criterion 4 and marked RECHECK in its `note`.

## Counts per source
| Source | Results returned | Duplicates skipped | Screened | Included | Excluded |
|---|---|---|---|---|---|
| Cited (related work) | 19 | 0 | 19 | 8 | 11 |
| Q1 FL IDS industrial IoT | 9 | 2 | 7 | 3 | 4 |
| Q2 FL IDS IoT dataset | 10 | 6 | 4 | 2 | 2 |
| Q3 FL IDS IIoT non-IID | 9 | 3 | 6 | 3 | 3 |
| Q4 FL IDS Edge-IIoTset | 9 | 4 | 5 | 4 | 1 |
| Q5 FL IDS TON_IoT | 10 | 8 | 2 | 0 | 2 |
| Q6 federated GNN IDS IoT | 10 | 4 | 6 | 3 | 3 |
| **Total** | 76 | 27 | **49** | **23** | **26** |

The target of 30 was not reached. All six queries were run to the end of their returned results (9 or 10 each).

## Exclusions by reason
| Reason | n | Records |
|---|---|---|
| 1: not peer reviewed (preprint only) | 2 | Q3-04, Q6-01 |
| 2: not an FL network-traffic IDS (survey, host/EDR logs, transaction data, host behaviour dataset) | 6 | C12 (EDR logs), C19 (transaction networks), Q1-05, Q4-04, Q5-09 (surveys), Q2-06 (host behaviour dataset) |
| 3: no public dataset | 1 | C15 (synthetic data only) |
| 4: full text not readable | 17 | Paywalled with no preprint: C09, C13, C14, C17, Q1-04, Q3-01, Q5-10, Q6-03, Q6-05. Citation could not be located: C06, C08, C10, C16. Access error, RECHECK: Q1-06, Q1-08, Q2-04, Q3-09 |

C12 also fails criterion 3, and C19 also fails criterion 4. Only the main reason is counted.

## Items to resolve before coding
- **RECHECK (4):** Q1-06 (Popoola et al., TCE 2024; accepted version on MMU e-space), Q1-08 (Discover IoT 2025, open
  access), Q2-04 (PMC12116512, open access), Q3-09 (Alsuwat, J. Big Data 2026, open access). These are probably
  readable when the sites are not rate-limiting or behind a CAPTCHA. If any turns out eligible, it should be
  re-screened in its place in the order. That would change which papers make the first 30, but with 23 included, all
  eligible papers are included anyway.
- **Cited references that could not be found:** C06 (Zhou et al., ESWA 299:130144), C08 (Mhawish et al., OJ-CS), C10
  (Purushottam et al., ICRTCST 2026) and C16 (Kumar et al., ICONAT 2025). Title and DOI searches did not find them.
  The manuscript's reference list should be checked.
- **Citation details that differ from the publisher:** C09 is article 19 (cited as 25(1)). C19 is article 111122.

## Addendum (2026-09-30, before coding)
C08 (Mhawish et al., IEEE Open Journal of the Computer Society, 2026, DOI 10.1109/OJCS.2026.3703977) was excluded
under criterion 4 because no copy could be found online. The authors hold the accepted author version (PDF, stored
as C08_Mhawish_OJCS2026.pdf). It meets criteria 1-4 and is included. Total included: 24.
