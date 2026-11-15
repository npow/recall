# Camera control is not character control

The released IF dataset has47third-person tasks out of50. Our native camera adapter always maps W/A/S/D to camera translation and arrow keys to rotation, irrespective of perspective. The current generator gets a fixed image caption, which sometimes explicitly says the subject is standing. Thus a requested character action is not necessarily represented in model conditioning.

The pinned native LingBot code explicitly sets `wasd_action=None`. Its six-channel camera rays become1536camera-projection features. The optional branch would append three extra channels and require2304features, which the released1536-feature checkpoint cannot accept as a switch. This is a capability/integration limit, not evidence that memory caused every IF failure. [Audit and source hashes](../reproduction/control-scope/audit.json).

A possible inference adapter would pre-encode action descriptions, keep one context stable through every noisy/clean pass of a chunk, and refresh cross-attention on text changes. Native inference passes an explicit `cross_attn_first_call`; clearing an `is_init` field alone does not refresh it. Holding the camera while a character moves could test this hypothesis on locomotion cases, but would not supply arbitrary gripper/button control. No such adapter is validated or promoted. Static caption/objective-text variants were already tested and failed the confirmation gate.

This limitation stays beside the full IF scores. It does not justify dropping tasks or substituting successful cases.
