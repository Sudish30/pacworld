# Owner updates

Short plain-language notes at each milestone: what happened, why, what it means, what's next. Newest first.
Anything that would need one of the hard limits (see the Workflow note in `notes/handoff.md`) is also written here.

## 2026-10-01 — Related work is written, the arXiv files can be built, and a robustness check is running

**What happened.**
- **Related work.** The paper's last missing section is written. Every cited paper was checked on arXiv, and the
  reviewer re-checked a sample and added three close papers.
- **What the literature check changed.** A context that mixes recent frames with sparser older ones is a known
  design in video models, so the paper no longer presents it as our idea. Other groups have also shown that video
  world models lose track of things they cannot see. What we did not find elsewhere: anyone measuring how well such
  a model keeps *time*, tying that to where its past frames sit, or comparing it with an "ideal observer" (the best
  any model could do with the same frames). The paper says "to our knowledge", since a search can miss things.
- **The failed "learn the controls from video" result is not new either.** A 2025 paper (Nikulin et al.) showed that
  this approach breaks when other moving things in the video (here: ghosts) distract it. Ours agrees with theirs.
- **arXiv files.** `bash tools/make_arxiv_bundle.sh` builds `paper/arxiv.tar.gz`. I have not submitted anything and
  will not; that step is yours.
- **Robustness check (running, about $2).** Each model in the "move one frame" follow-up was trained once. Training
  has randomness (the "seed"), so one run could be luck. I registered the rules in advance and started 18 more runs
  with new seeds: 4 each for the two headline models, 2 each for five others.

**One mistake to report.** My script for starting the pod had a bug and started two. I deleted the extra one within
minutes; it cost a few cents.

**What's next.** When the 18 runs finish: score them by the registered rules, have the reviewer check, update the
README and paper whichever way it comes out, rebuild the arXiv files, and tell you they are ready.

## 2026-10-01 — The reviewer is set up; its first review found real errors, now fixed and pushed

**What happened.** I created the independent reviewer (`.claude/agents/research-reviewer.md`) and ran it on the README
and paper that were pushed earlier today. It checked every number against the result files. Verdict: approve with
changes. I verified each of its points against the records, applied them, had the changes re-checked, and pushed.

**What it found** (all fixed):
- **One factual error.** I had written that oversampling the end of the "frightened" phase (the period when ghosts
  turn blue and can be eaten) made both models end the phase too early. That was true only for the model with the
  longer reach (reach = how far back in time its oldest context frame sits). The other model let phases run on.
- **One registered consequence I had softened.** The DIAMOND test's pre-registration said that a failure "counts
  against the rule". The README called it uninformative instead. It now states the registered outcome first and
  labels our "the ghosts vanish, so it says little" argument as post-hoc (made after seeing the result).
- **One overclaim.** "Every experiment was pre-registered" was too strong: the first diagnosis was exploratory, and
  the context comparison only had an informal criterion, part of which the strided model missed (longest pen stay 148
  against a target of about 91).
- **Missing disclosures.** Some ghost-count numbers rest on only 3-4 of the 10 test episodes. Two statistically
  significant differences that go against our preferred model were not listed. Each model was trained only once
  (one "seed", the random starting point of training), so some differences could be luck.

**What it means.** The main findings stand; the write-up is now more careful about what was predicted in advance and
how much data each number rests on. One note on the setup: the reviewer agent type only loads when a session starts,
so in this session a stand-in agent was given the same instructions. From the next session it loads directly.

**What's next** (the reviewer's ranking):
1. Write the related-work section and prepare the arXiv files (no GPU cost). You submit; I only prepare.
2. Repeat the follow-up experiment with new training seeds (about $2), to check that "moving one frame fixes it" is
   not a one-off. This needs its own pre-registration first.
3. If money remains: more test episodes for the Ms. Pac-Man comparison (about $2-3).

Budget: balance $21.01, of which about $18 is usable (it must stay above $3).

