# Start-view re-anchoring replay

Six matched H100 native-control episodes are complete: GC027, seeds42/43,
stock18/6, four-view memory, and the existing start-view re-anchor. Each replays
WAIT then12RIGHT chunks at30degrees/second, without an adaptive player.

Claude means: stock3.80, four-view4.00, start-anchor3.75. Every trajectory gate
is partial and reports no visual revisit. Endpoint inspection also finds no
restoration of the initial sports-center sign/view; see endpoints.jpg and the
complete videos. Earlier access to exact initial-view keys alone is not enough.

Gemini rescoring is in reproduction/gemini-current/. Keep its scores separate
from Claude and from published overall scores; a high consistency rating is not
proof of a completed360-degree visual return.

All six runs passed13/13 native camera checks and persistent-worker cleanup.
Two constructors served six episodes. Generation failures would have remained
in the summary; none occurred. This is an exploratory replay of a known failure,
not a held-out benchmark or a new successful method.
