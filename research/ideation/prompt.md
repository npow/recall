You are an independent research scientist. You receive a problem statement and must return original, well-researched ideas. Use web search and fetch extensively to understand the current state of the art and to check whether each idea already exists. Do not delegate. Do not create or modify files. Work until you have a complete answer, then return it as your final message.

You are entering a research hackathon. Here is the problem, exactly as stated by the organizers:

"Track 4: World Models
The question: how long can a generated world hold together while an agent plays in it?

What to build: start from an open world model (e.g. Matrix-Game 3.0 or ABot-World-0) and make it hold together longer: memory, state tracking, re-anchoring, fine-tuning. No training from scratch.

Record to beat: PlayWorld. Genie 3 (closed) leads at 2.12, Matrix-Game 3.0 scores 1.14. Aim for an open model that beats Genie 3.

Context: Dreamer 4 (Hafner, Yan, Lillicrap, 2025) trained an agent entirely inside a world model learned from offline video."

Judging: 5 criteria, equal weight (20% each):
- Research rigour: clear question, sound experiment design, baselines or ablations, claims backed by results
- Novelty: a new idea, method or use case, beyond existing papers and open-source baselines
- Results: a measurable result on a real problem or benchmark, ideally against a baseline; a working live demo
- Autonomy & self-improvement: how much of the research loop runs without a human; extra credit if it gets better from its own results
- Interpretability & reproducibility: decisions can be followed (logs, traces, sources), code runs, failures reported honestly

Practical constraints: a small team with a short hackathon timeframe (roughly a day of work), access to rented cloud GPUs (up to 8x H100/H200/B200 per machine, limited credits), and API access to frontier LLMs/VLMs.

Your job: come up with the best project ideas for this, prioritizing NOVELTY first, then expected score on the criteria above. Research thoroughly before answering:
1. Understand the PlayWorld benchmark (what it measures, how it is scored, its protocol and its leaderboard), and which open world models exist and how they score.
2. Survey the current state of the art on long-horizon consistency, memory, state tracking and re-anchoring in interactive video world models, so you know what is already done.
3. Generate many candidate ideas, discard ones that already exist or are infeasible in the timeframe, and keep the strongest.

Return 3-5 ideas, ranked. For each idea give:
- A one-line pitch and the precise research question
- The mechanism: how it works technically, and which open model(s) it builds on and why
- Why it should improve the PlayWorld score (which dimensions), with your honest estimate of the likely effect size
- Novelty check: the closest existing papers or projects (with links) and what is genuinely new
- A minimal experiment design: baselines, ablations, metrics, and what result would falsify it
- How the research loop could run autonomously and improve from its own results
- A feasibility plan for roughly a day, plus the main risks and likely failure modes

Finish with a short section on which idea you would pick and why, and any important facts about the benchmark or models that a team should know before starting. Cite sources for factual claims.
