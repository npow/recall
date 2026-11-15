| Case | stock Gemini (Opus) | cache46 Gemini (Opus) | fb Gemini (Opus) | fbcomp Gemini (Opus) |
| --- | ---: | ---: | ---: | ---: |
| GC005 | 1.00 (1.00) n=2 | 1.00 (1.80) n=1 | 1.00 (2.40) n=2 | 1.00 (–) n=2 |
| GC007 | 3.30 (1.40) n=2 | 4.30 (1.00) n=2 | 1.40 (3.20) n=2 | 4.00 (1.00) n=1 |
| GC015 | 1.90 (2.75) n=2 | 2.30 (3.30) n=2 | 1.00 (2.40) n=1 | 1.00 (1.90) n=2 |
| GC017 | 1.60 (3.80) n=2 | 2.20 (–) n=2 | 4.00 (3.35) n=2 | 1.00 (4.00) n=1 |
| GC019 | 1.30 (2.40) n=2 | 2.30 (2.00) n=2 | – | – |
| GC024 | 3.10 (3.15) n=2 | 4.80 (3.20) n=2 | 3.70 (2.80) n=2 | 3.10 (3.30) n=2 |
| GC025 | 5.00 (3.70) n=2 | 5.00 (4.00) n=1 | – | – |
| GC027 | 2.00 (3.10) n=2 | 3.60 (2.40) n=2 | 1.60 (–) n=1 | 3.10 (2.60) n=2 |
| OE026 | 1.38 (2.00) n=2 | 1.75 (1.75) n=2 | – | – |
| OE029 | 1.42 (2.00) n=2 | 1.33 (2.42) n=2 | – | – |
| OE044 | 1.00 (1.75) n=2 | 1.00 (1.75) n=2 | – | – |
| OE055 | 1.00 (–) n=1 | 1.00 (1.88) n=2 | – | – |
| OE056 | 1.00 (–) n=2 | – | – | – |
| OE057 | 1.00 (1.75) n=2 | 1.00 (2.00) n=1 | – | – |
| OE069 | 1.38 (1.88) n=2 | 3.00 (2.00) n=2 | – | – |

GC dimension means (rubric called only when the gate did not fail)
| Arm | n | identity_id | identity_texture | color | relative_position |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock | 15 | 2.20 | 2.53 | 3.20 | 2.33 |
| cache46 | 13 | 3.00 | 3.62 | 4.31 | 3.00 |
| fb | 10 | 2.00 | 2.30 | 3.20 | 1.90 |
| fbcomp | 8 | 2.50 | 2.25 | 2.38 | 2.50 |

Out-of-sight OE dimension means
| Arm | hidden_state_progression | out_of_sight_evolution | physical_causality | process_accumulation | prompt_evolution_alignment | revealed_state_difference | subject_identity | temporal_scene_consistency | trajectory_continuity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stock | 1.29 (n=7) | 1.50 (n=2) | 1.00 (n=2) | 1.14 (n=7) | 1.50 (n=2) | 1.43 (n=7) | 1.11 (n=9) | 2.00 (n=2) | 1.00 (n=2) |
| cache46 | 2.00 (n=6) | 1.50 (n=2) | 1.00 (n=2) | 2.00 (n=6) | 1.00 (n=2) | 2.00 (n=6) | 1.88 (n=8) | 1.00 (n=2) | 1.00 (n=2) |

Gate verdict counts per arm (Gemini)
stock {'fail': 5, 'partial': 9, 'pass': 15}
cache46 {'fail': 4, 'partial': 10, 'pass': 11}
fb {'partial': 9, 'pass': 1}
fbcomp {'fail': 2, 'partial': 6, 'pass': 2}
v3stock {'partial': 1}
