"""Experimental text-prompt and translation-gain controls for LingBot fixed-path diagnostics.

Two default-off protocol deviations, used only to ask whether the model can be made to carry
out commanded camera motion (forward translation barely moves the camera in stock runs):

* ``--motion-prompt static``  appends one motion sentence to the task caption for the whole rollout.
* ``--motion-prompt perchunk`` appends a sentence matched to each chunk's commanded action. The
  per-action contexts are encoded once (while the text encoder is still on the GPU), and at the
  first denoiser call of a chunk whose prompt differs from the previous chunk's the cross-attention
  K/V cache is recomputed (``cross_attn_first_call=True``), exactly the mechanism upstream uses
  once per generate.
* ``--translation-gain g`` multiplies the (already max-normalised) per-frame translations that
  reach the camera embedding. Upstream divides translations by the sequence maximum, so a uniform
  rescale of the poses cancels; only a post-normalisation gain changes what the model sees.
"""

STATIC_SENTENCE = "The camera moves steadily forward through the scene, advancing toward what lies ahead."

ACTION_SENTENCES = {
    "W": "The camera moves steadily forward, getting closer to the scene ahead.",
    "S": "The camera moves steadily backward, pulling away from the scene.",
    "A": "The camera slides steadily to the left.",
    "D": "The camera slides steadily to the right.",
    "LEFT": "The camera turns steadily to the left.",
    "RIGHT": "The camera turns steadily to the right.",
    "UP": "The camera tilts steadily upward.",
    "DOWN": "The camera tilts steadily downward.",
    "WAIT": "The camera stays still.",
}


def prompt_for(caption: str, mode: str, action: str | None) -> str:
    """Prompt text for one chunk. ``off`` returns the caption unchanged (stock)."""
    if mode == "off":
        return caption
    if mode == "static":
        return f"{caption} {STATIC_SENTENCE}"
    if mode == "perchunk":
        return f"{caption} {ACTION_SENTENCES[action]}"
    raise ValueError(f"unknown motion-prompt mode: {mode}")


class PromptSwitcher:
    """Chooses the text context for each denoiser call and flags the first call after a prompt change."""

    def __init__(self, caption: str, mode: str, actions: list[str]):
        self.caption, self.mode = caption, mode
        self.texts = {}
        for action in set(actions) | {"WAIT"}:
            self.texts[action] = prompt_for(caption, mode, action)
        self.contexts = {}
        self.previous_text = None
        self.switches = []

    def encode(self, text_encoder, device):
        """Encode every distinct prompt once; the encoder must still be resident on ``device``."""
        for text in sorted(set(self.texts.values())):
            self.contexts[text] = [tensor.to(device) for tensor in text_encoder([text], device)]

    def for_chunk(self, index: int, action: str):
        """Return (context list, force_first_call, text) for the first denoiser call of chunk ``index``."""
        text = self.texts[action]
        changed = self.previous_text is not None and text != self.previous_text
        if changed:
            self.switches.append({"chunk": index, "action": action, "prompt": text})
        self.previous_text = text
        return self.contexts[text], changed, text

    def current_context(self, action: str):
        return self.contexts[self.texts[action]]


def gain_compute_relative_poses(original, gain: float):
    """Wrap ``compute_relative_poses`` so translations are scaled by ``gain`` after normalisation."""
    if gain == 1.0:
        return original

    def wrapped(*args, **kwargs):
        poses = original(*args, **kwargs)
        poses[:, :3, 3] = poses[:, :3, 3] * gain
        return poses
    return wrapped
