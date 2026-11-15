# Plain-language explainer: what we built and what happened

For the presenter. The project is called RECALL (Retrieval of Earlier Camera views for Long-horizon Lookback); the code package is still named `worldsync`. Every term used in the video and slides is defined here. Numbers are from
[LEADERBOARD.md](../LEADERBOARD.md); method details are in [METHOD.md](METHOD.md).

## 1. The model and why it forgets

LingBot-World2 generates video a few frames at a time. Each step makes one **chunk**: 4 **latent frames**
(a latent frame is the model's compressed version of about 4 video frames, so a chunk is about 0.75-1 s of video).

To make the next chunk, the model looks back at earlier latent frames through **attention**. It cannot look at all
of them, so it keeps a fixed-size memory called the **KV cache**: for each remembered latent frame it keeps the
"keys and values" (K/V), the internal numbers attention reads, at each of its 40 layers.

The stock cache holds **18 latent frames**, split into two parts:

- **Sink 6**: the *first* 6 latent frames of the episode, kept forever. They anchor the overall look of the scene
  (this is why the model never completely loses the starting style).
- **Recent 12**: the 12 *most recent* latent frames, a sliding window. When a new chunk arrives, the oldest
  recent frames are thrown away ("evicted").

18 latent frames is about **4.5 seconds**. Anything the camera saw more than about 4.5 s ago, other than the very
first moment, is gone. When the camera turns away and comes back later, the model has nothing to look at for that
place, so it invents a new one: a different building, a re-lettered sign, a missing object.

## 2. Our episodic memory, step by step

Idea: keep the old frames somewhere else, and hand the right ones back to the model when the camera returns.

1. **Archive.** After each chunk is generated we save its latent frames, together with the **camera pose** at that
   moment (where the camera is and which way it faces, tracked from the movement commands) and a small visual
   fingerprint of the picture (a DINO descriptor).
2. **Detect a return.** Before generating a chunk we look at the pose the camera is about to have and ask:
   have we been here before, facing this way, longer ago than the 4.5 s window?
3. **Retrieve.** If yes, pick up to K = 10 archived latent frames of that place.
4. **Give them to the model.** For that chunk, the attention window becomes
   **sink 6 + retrieved 10 + recent 8** instead of sink 6 + recent 12. The model's weights are not changed and
   nothing is trained; it simply gets to look at its own earlier frames of that place again.

If nothing qualifies in step 2, we do nothing, and the output is exactly the stock model's (pixel for pixel).

### "Pose-verified" retrieval

The first version picked the 10 archived frames with the nearest camera pose, always. Two problems showed up in our
audits: it swapped old frames in even when the camera had not returned anywhere (on 47% of non-revisit chunks), and
the commanded pose can be wrong: if the camera is told "forward 4, back 4" the model often does not visually undo
the forward motion, so the frame stored for that pose shows a different picture.

**Pose-verified** means a stored frame is used only if we have reason to trust that the pose match is a real return:

- either only **rotation or waiting** was commanded since that frame was stored (the model follows rotations
  accurately: about 177 degrees for a commanded 180), or
- the **picture confirms it**: the current view looks like a stored view from that place (DINO similarity above a
  threshold).

Otherwise nothing is retrieved. In an offline replay of 726 real chunks this retrieved at 108 of 115 real revisits
and at 0 non-revisits (the old rule: 115 and 338).

### "Recompute" and "splice"

What attention needs from a remembered frame is its K/V at all 40 layers, about **1.2 GB per latent frame**. The
first version stored that, so it could afford only 24 archived frames (about 31 GB of RAM), and old ones were dropped.

- **Recompute**: store only the small latent frame itself (about **6 MB**, roughly 200x less) and, when a frame is
  retrieved, run it through the model once (a "clean pass", the same pass the model already runs after every chunk)
  to rebuild its 40 layers of K/V. Nothing ever has to be dropped. On the test path this gave the same recall as
  storing the K/V (0.77 against 0.78-0.82; stock 0.35). This was the presenter's idea.
- **Splice**: put those rebuilt K/V into the attention window, between the sink frames and the recent frames, with
  the right position labels so the model treats them as earlier frames of the same video.

### Where "200x less storage" comes from

Two ways to remember one latent frame:

- **Cache its K/V** (what the model itself does for the 18 frames in its window, and what memory v1 did):
  40 layers x 1,560 image patches x 5,120 numbers per patch x 2 (keys and values) x 2 bytes each
  = about 1.28 billion bytes. Measured: **1,235 MB per latent frame**.
- **Store the frame and recompute** (memory v2): keep the frame's compressed latent plus the small inputs needed to
  run it through the model again (its image condition and camera embedding). Measured: 308 MB for 50 latent frames
  = **6.2 MB per latent frame**.

1,235 / 6.2 = **about 200x**. The trade is compute for storage: when a frame is retrieved we spend one extra
forward pass to rebuild its K/V (13.5 s in total for 22 frames in our test job), and we store 200x less. For a
40-chunk episode (160 latent frames) that is about 1 GB instead of about 200 GB, so nothing has to be evicted.

It is the same information: the K/V are a deterministic function of the frame and its inputs, so they can be
rebuilt instead of kept. They come out slightly different from the originals, because the original K/V also
depended on what each frame was attending to at the time; in our test the two versions retrieved the same frames
and scored within 0.05 of each other (one scene, one seed).

What did *not* give big savings (our ablation): compressing the stored K/V. 8-bit storage halves it and keeps
quality; using only some layers, pooling patches, or low-rank compression lost the effect on a second scene.

## 3. Which version produced which result (important for questions)

| Name | What it is | Where it was tested |
|---|---|---|
| Memory v1 ("budget memory", exact storage) | stores K/V for up to 24 frames, nearest-pose retrieval | full 171-task benchmark (no gain); **revisit-horizon benchmark (the positive result)** |
| Memory v2 (disk or recompute) | archives everything, pose-verified retrieval, recompute | scripted paths (works, removes v1's harm); 36 real revisit tasks (not better than stock) |
| Director | an LLM writes event captions for the model's text prompt | full 171-task benchmark (+0.02, no reliable gain) |

Note on window size: the attention window with memory is sink 6 + retrieved 10 + recent 8 = 24 latent frames, larger than stock's 18, so part of the +0.77 may come from the larger window rather than from retrieval (not isolated; a 46-frame stock cache reached about 0.82 on the scripted diagnostic).

The slides and video now present the first design as "our method" (it produced the +0.77) and the store-and-recompute idea as a separate storage finding (scripted path only). The earlier text: the video's architecture diagram describes v2, and the headline chart (+0.77) was measured with v1 using exact
storage. Both are "our episodic memory"; say so if asked. v2 was not in that benchmark because it had not passed its
tests when the benchmark was frozen.

## 4. What happened, in order

1. **Reproduced the baseline.** Stock LingBot-World2 on all 171 PlayWorld tasks: 1.85 (Modal) and 1.81 (Lambda)
   against the published 1.82.
2. **Built memory v1.** On scripted "turn away, wait, turn back" paths it brought the camera back to the start
   (similarity 0.30 -> 0.78). On the full benchmark it did nothing: 1.80 against 1.85.
3. **Found out why** (independent audits): PlayWorld episodes are short (median 5-8 s), so the stock 4.5 s window
   already covers most of them; only 12% of chunks are real revisits; v1 also had three bugs and two design flaws.
4. **Tried a text director** to make scenes evolve: a small positive trend, not established (+0.02 primary,
   +0.07 on three judge calls). Record 2.12 not reached.
5. **Built memory v2** from the audit findings (pose-verified, recompute). Fixed v1's harm and cut storage 200x.
   On 36 real PlayWorld revisit tasks it was not better than stock (-0.35 on three judge calls).
6. **Ran a pre-registered revisit-horizon benchmark**: 20 unseen PlayWorld scenes, look away for 0/6/12/24 s and
   turn back, stock against memory, PlayWorld's judge. Memory is better at every length: pooled +0.77
   (95% CI +0.51 to +1.04). This is the clear positive result.

## 5. One-sentence answers to likely questions

- *Did you beat Genie 3?* No. Full benchmark 1.80-1.86 against 2.12; we matched the open baseline.
- *Then what improved?* Consistency after the camera looks away: +0.77 on a 1-5 judged scale, 20 unseen scenes,
  intervals above zero at every absence length.
- *Why not on PlayWorld's own tasks?* Its episodes are mostly shorter than the model's own memory, and most low
  scores come from frozen scenes or unattempted actions, which memory cannot fix.
- *Is it trained?* No. No weights change. A trained memory LoRA was tried and made generation worse.
- *What does it cost?* About 6 MB per remembered frame; a chunk where memory fires takes several seconds longer.
- *How noisy is the judge?* The same video scored three times varies by about 0.36; identical videos have
  received 1.0 and 5.0. We average calls and pair runs by scene and seed.
