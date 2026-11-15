# WorldSync — two-minute voiceover

The MP4 is captioned and silent. Read this script over it, or use it as a live presentation guide.

**0:00–0:10 — Hook**

This is WorldSync. We’re exploring how to make an open, generated world follow through on what a player asks it to do—without retraining the world model.

**0:10–0:25 — Method**

We start with LingBot-World2. Our configuration gives the generator the task itself, maintains camera state across chunks, and uses visual feedback to adjust the movement schedule. The model’s weights stay frozen.

**0:25–0:42 — Quantified result**

On three preselected instruction-following tasks from PlayWorld, the average score rises from one point five six to two point four two: a gain of zero point eight six. All three tasks improve. Each video gets three Gemini judgments.

**0:42–1:12 — Boardwalk comparison**

Here’s the biggest gain. Both runs start with the same boardwalk scene. The legacy player ends its episode after about eight seconds; that panel then holds its final frame. WorldSync continues for a full minute, shown here at double speed. Its mean score rises from one point six seven to three point five. This is promising scene continuity, though the instruction to keep walking is still only partially completed.

**1:12–1:28 — Pond comparison**

The pond task improves too: two point zero to two point five six. This excerpt shows the first sixteen seconds. The character progresses toward the pond, although later frames still develop texture artifacts.

**1:28–1:44 — Separate memory example**

We also tested archived-view memory separately. After turning away and waiting, we retrieve earlier attention features to recover familiar landmarks. On this controlled revisit, endpoint visual similarity rises from zero point five seven four to zero point seven four six.

**1:44–2:00 — Close**

These are promising, concrete results from a two-day project: better scores on a small task subset, and a working memory mechanism. We haven’t won the full benchmark. Next, we want to combine these ideas and make evolving processes more consistent.
