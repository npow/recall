# Voice-over script for docs/demo/presentation.mp4

One paragraph per scene, in scene order. `scripts/build_voiceover.py` reads this file, synthesises each paragraph locally
(Piper, no online service), fits it to the scene's length and muxes the track into the video. Numbers match the on-screen text.

## title
RECALL: a training-free episodic memory for an open world model.

## problem
Generated worlds forget what they leave behind. Look away, then turn back, and the place has changed. How long can a world hold together?

## benchmark
In PlayWorld, an AI agent plays inside the generated world and a judge scores it. The record to beat is Genie 3's two point one two.

## failures
We reproduced the baseline: one point eight one and one point eight five, against the published one point eight two. The released model breaks: rooms are rebuilt, boats never leave, wagons appear from nowhere.

## insight
The model only remembers four and a half seconds. Turn away longer and that view is gone.

## system
Our memory saves each frame's cached keys and values with its camera pose, retrieves the ten nearest by pose, and splices them into attention. No training, and the weights are untouched.

## timeline
Old frames leave the cache as the camera turns away. On the way back the nearest saved ones flow into attention. Our window is twenty-four frames, stock's eighteen.

## hero
Same scene, same camera path, same time away. As released, the returning view is a different place. With our memory, the storefront it started from is back.

## chart
Consistency improves at every length: plus zero point seven seven, interval zero point five one to one point zero four.

## ablation
Store the frame and recompute: two hundred times smaller, similar recall on one scripted path.

## real
Real task: the released model invents a traffic light.

## limits
We did not beat Genie 3. On PlayWorld's own tasks we score about one point eight, against two point one two, and verified retrieval was not better. Frozen scenes and a noisy judge explain much of the rest.

## process
AI agents ran the loop: propose, run, audit, fix, re-test. A human set goals and budget.

## future
Next: a trained memory layer and retrieval by what is visible. Code and logs are in the repo.
