# Learned reader video diagnostic

Both 14-chunk GC010 seed42 videos completed on H200 with the same frozen controls and settings. The first four chunk pixel hashes are identical; the first actual four-frame retrieval occurs at chunk4, when the reader can first affect generation. See [summary](summary.json), [sampled frames](contact.jpg), full videos and run manifests in the arm directories. The trained checkpoint hash matches the held-out training report.

Both final sampled views recover recognizable parts of the original street, with framing and lettering drift. This review does not demonstrate a quality gain from the reader, and does not certify full-path continuity. No Gemini score is assigned. The prior strict training gate failed on one of two held-out sources; this additional diagnostic does not override that failure or promote the reader. The hard pose archive is still required: this is a learned reader, not a learned write-head or a reproduction of Titans/NTM.

The frozen plan and jobs precede generation. An initial missing-job-file launch failed before generation and is retained in `failed-startup/`. Both valid jobs completed; total parent wall time includes cloud queue/startup. No core inference default is changed.
