# Why a failed return can score 3.8

This is a limitation of the released gate/rubric combination in these two episodes, not an arithmetic error. Replaying the pinned released `normalized_score` on both saved judgments reproduces exactly1.2 and3.8.

The [instruction gate](https://github.com/kxding/PlayWorld/blob/eeaea12e050a90b815d11d030e943e7b3181da21/metrics/vqa/instruction_gate.py) explicitly maps both PASS and PARTIAL to1. It defines partial as meaningful progress with an incomplete trajectory. The [aggregator](https://github.com/kxding/PlayWorld/blob/eeaea12e050a90b815d11d030e943e7b3181da21/metrics/aggregate.py) admits that rubric score whenever the numeric gate equals1; it does not additionally require `revisit_observed=yes`.

For the70-frame video, the judge reports no revisit but gives object structure4 (double weight), texture/text1, color5 and relative positions5. The released GC weighting therefore produces `(2*4 + 1 + 5 + 5)/5 = 3.8`. Task-specific completion is excluded from those categories. The model can receive credit for a coherent-looking portion of a failed journey.

Consequently **3.8 does not establish long-horizon memory, task completion or a successful revisit**. The contact sheet and full videos show no return to the initial sports-center view. Judge noise and perceptual mistakes remain additional limitations; matching the released arithmetic does not prove every visual judgment correct.

Keep the released score for benchmark comparison. Report trajectory completion and actual revisit evidence beside it, and require an actual verified return before claiming improved revisit memory. Changing the gate to reject partial trajectories would define a different metric and cannot be compared directly with published1.82/2.12 scores.

The wrapper tests the verdict string while the released aggregator checks the numeric gate. An [audit of55 saved top-level gate reports](../completion-audit.json) found no contradictions between those fields. It also identified three reports with scores at least3 despite no judge-reported revisit. These counts refer to saved reports, not unique benchmark cases or independently verified successes. Reproduce with `scripts/audit_completion.py`.
