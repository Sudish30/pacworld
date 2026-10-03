# Owner updates

Short plain-language notes at each milestone: what happened, why, what it means, what's next. Newest first.
Anything that would need one of the hard limits (see the Workflow note in `notes/handoff.md`) is also written here.

## 2026-10-02 — The paper is prepared for arXiv

**What changed.** An Acknowledgments section with the AI-use disclosure (your wording, adjusted in one place for
accuracy: within your rules, the AI reviewer approved designs and results and chose some next steps). A shorter
abstract (about 250 words) that says what the problem was, what fixed it, the one-frame result and its replication,
and that several tests failed. The detailed caveats moved to the Limitations section; none was dropped. Plainer
sentences throughout. "Draft" is gone from the title page.

**What did not change.** No number, claim or result. A script compared every number in the old and new text; the
only differences are numbers that left the abstract and still appear in the body. The reviewer checked that every
caveat survived and caught one slip I introduced while rewording (a sentence that blamed the strided model for a
difference that belongs to the 8-frame model); it is fixed.

**Where the files are.** `bash tools/make_arxiv_bundle.sh` rebuilds `paper/arxiv.tar.gz`; the PDF is
`paper/main.pdf`. Submitting is yours.

**Explain it back.** *Why does arXiv ask authors to disclose significant AI use, and what does our disclosure say?*
<details><summary>Answer</summary>
Readers and reviewers need to know who did what to judge the work. Ours says that AI assistants wrote code, ran
experiments, reviewed designs and drafted the text; that you set the goals and rules, made the main decisions and
take responsibility; and that within your rules an AI reviewer approved designs and results. It also points to the
repository, where every pre-registration and review is logged, so anyone can check.
</details>

## 2026-10-02 — The arXiv files are ready, the video is made, and the planned work is done

**arXiv files: ready for you.** Run `bash tools/make_arxiv_bundle.sh`; it writes `paper/arxiv.tar.gz`, which is what
arXiv's upload form takes (the paper's source, its reference list and its one figure). I checked that it compiles on
its own. I have not submitted anything; that step is yours. Read the PDF first (`cd paper && ../.venv/bin/tectonic
main.tex` makes `paper/main.pdf`): it carries your name.

**Video:** `docs/pacworld_60s.mp4`, 60 seconds, silent, sized for LinkedIn.
- 0-4 s: title.
- 4-34 s: the real game beside the 4-frame model and the strided model, same start and key presses. A yellow box
  marks the ghost pen, and a counter under each panel shows how long the pen has been occupied (it turns red past the
  real game's maximum of 91 steps). The 4-frame model's counter runs past 300.
- 34-54 s: the one-frame experiment as a bar chart: 37, 43, 45, 49, 47 parked out of 60 with the frame 81 steps back;
  0 in all five with it exactly 80 back.
- 54-60 s: the demo link.
The reviewer made me add, on screen, that the rollout shown is a picked example and what the real rates are (17 of
30 against 0 of 30), and that the 80-step fix was proposed after the first run and registered before it was tested.

**Money.** Balance $16.21. The autonomous run spent about $4.80. No GPU is running, and the reviewer's decision is
to stop experiments here and keep the rest in reserve.

**The work is done.** Nothing else is planned or running. What is left is yours: submit to arXiv if you are happy
with the PDF, and post the video.

**Explain it back.** *In the video's bar chart, why are there five bars per row instead of one?*
<details><summary>Answer</summary>
Each bar is a separate model, trained from scratch with a different random seed. One bar could be luck. Five bars
that all agree (many parked ghosts with the frame 81 steps back, none with it exactly 80 back) show the effect comes
from where the frame sits, not from the luck of one training run.
</details>

## 2026-10-02 — The main Ms. Pac-Man result held up in a second training run; one smaller claim did not

**What happened.** The main result (older frames in the model's view stop it from parking ghosts) came from a single
training run of that model. I trained it again from a different random starting point (a new "seed"), with the
pass/fail rule fixed in advance. Cost: about $2.80.

**Result: it replicates.** The new run also parked ghosts in 0 of 30 test rollouts, against 17 of 30 for the model
that sees only its last 4 frames. The registered rule needed the difference to be clearly below zero and at most 3
parked rollouts; it got −0.57 [−0.77, −0.37] and 0.

**Two honest caveats.**
- **The timing is not exact.** The new run holds ghosts for a typical 104 steps and at most 138; the real game holds
  them 78 and at most 91. Parking is gone, precise timing is not.
- **One smaller claim is withdrawn.** I had reported that the same change also made Pac-Man's position more accurate
  (by about 5 pixels). In the new run it did not (1 pixel, well within the noise). That earlier number was one run's
  luck, so the README and paper now say there is no reliable evidence for it. This is exactly what a second seed is
  for.

**What it means.** The headline is on firmer ground: the parking fix has now been seen in two independent training
runs on the real game and in five on the small test game. GPU experiments stop here on the reviewer's decision.
About $13 of usable balance stays in reserve.

**What's next.** Rebuild the arXiv files and tell you they are ready, then the 60-second video.

**Explain it back.** *Why did we train the same model twice, and what did the second run change?*
<details><summary>Answer</summary>
Training starts from random numbers and feeds the data in a random order, so two runs of the same recipe give
slightly different models. If a result only appears in one run, it may be luck. The second run kept the important
result (no parked ghosts) and removed an unimportant one (better Pac-Man position), which turned out to be luck. A
result seen in two runs is much more believable than one seen in one.
</details>

## 2026-10-02 — The demo is live: https://sudish30.github.io/pacworld/

**What happened.** The served model now runs in a visitor's browser, with no server and no cost. It is linked at the
top of the README.

**The pilot (the cheap test first).** The bar was written down before measuring: at least 10 frames per second in
Chrome on your Mac, and frames that match the PyTorch model closely.
- **Speed: 13.8 frames per second** (the game's own rate is 15, so it plays at about 92% speed).
- **Accuracy: the browser's frames differ from PyTorch's by 0.10 on the 0-255 colour scale on average**, at most 2,
  on all 16 test cases. The bar was 1.0.
- Both passed, so I built the page.

**Three things had to be solved.**
1. **Half-size numbers.** To halve the download (39 MB instead of 76), the weights are stored as 16-bit numbers
   ("half precision") instead of 32-bit. The reviewer spotted that one part of the model, the part that tells the
   network how noisy its input is, needs more precision than 16 bits give. I compute that part once, in full
   precision, and store the result inside the model file.
2. **A bug in the browser's GPU library.** The first run was fast but drew wrong frames. The same model was exact on
   the browser's CPU path, so I compared the two paths layer by layer. The very first layer was wrong on the GPU: it
   mishandles an input with 33 channels. Padding the input to 36 channels with zeros fixes it and changes nothing
   mathematically.
3. **Where to put a 39 MB file.** It is attached to a GitHub "release" instead of being committed, so the repo's
   history stays small. A small script on GitHub builds the site from the repo plus that file.

**What the page says.** Plain English: what you are looking at, what to watch for (the ghost pen), the finding, and
an honest-limits list (it drifts, ghosts fade, it was trained once, 64x64, not affiliated with the game's owners).
The reviewer rewrote 12 sentences to remove overclaims.

**Two things for you to know.**
- **Intellectual property.** The page shows frames of, and a model trained on, Ms. Pac-Man, which belongs to Bandai
  Namco. For a small non-commercial research demo the risk is low and similar demos exist, but a takedown request is
  possible. To take the demo down yourself: repo Settings -> Pages -> disable, and delete the release `web-demo-v1`.
- **Phones do not work.** It needs a keyboard and WebGPU (desktop Chrome or Edge). Other visitors see a recording.

**Explain it back.** *Why can the model run in a browser at all, and why did we store the weights as 16-bit numbers?*
<details><summary>Answer</summary>
The model is small (18.8 million numbers) and each frame needs only three passes through it, so a laptop's graphics
chip can do it about 14 times a second. Browsers can now use the graphics chip through a feature called WebGPU.
Storing each weight in 16 bits instead of 32 halves the download and barely changes the output (0.10 out of 255),
because the network's arithmetic does not need more precision than that, except for one small part, which we
computed ahead of time in full precision.
</details>

## 2026-10-02 — The robustness check: the headline result holds up; one side claim does not

**What happened.** The 18 extra training runs finished (about $2). Each repeats an earlier experiment with a
different "seed", the random starting point of training, to see whether the result was luck.

**What it showed.**
- **The headline holds, 4 out of 4.** With a past frame 81 steps back, the model fails to time an 80-step wait in
  every new run (it leaves the ghost parked in 43-49 of 60 tries). With the frame exactly 80 steps back, it never
  parks, in every new run. This was the registered test, and it passed.
- **A side claim failed.** I had written that short gaps between past frames were harmless, because two short-gap
  settings passed the first time. The registered check on this failed: one of them (a 24-step wait) parks when
  trained again with new seeds. Its first pass was luck. Only the very shortest gap (8 steps) is reliably fine.
- **Net effect on the story.** It is simpler and a bit stronger: when the hidden wait falls between two of the
  model's past frames, the model tends to park, almost everywhere we tested. That is closer to your original
  prediction than to the "ideal observer" calculation (the best a model could do in theory), which expected the
  mid-gap cases to mostly work.

**The numbers** ("parked" = tries, out of 60, in which the model never released the ghost):

| the 80-step wait, past frame placed | first run | new run 1 | new run 2 | new run 3 | new run 4 |
|---|---|---|---|---|---|
| 81 steps back (one step off) | 37 parked | 43 | 45 | 49 | 47 |
| exactly 80 steps back | 0 parked | 0 | 0 | 0 | 0 |

| other waits that fall between two past frames (share of tries released; higher is better) | first run | new run 1 | new run 2 | needed |
|---|---|---|---|---|
| 72 steps | 0.78 | 0.63 | 0.67 | 0.80 |
| 79 steps | 0.35 | 0.30 | 0.23 | 0.49 |
| 88 steps | 0.55 | 0.60 | 0.40 | 0.77 |
| 8 steps (very short gap) | 1.00 | 0.98 | 0.97 | 0.86 |
| 24 steps (short gap) | 0.82 | 0.73 | 0.53 | 0.79 |

So: 8 of the 18 new runs tested the headline and all 8 agreed with it. The other 10 tested "long gaps park, short
gaps are fine": the long-gap half held in all 6 runs, the short-gap half failed (the 24-step wait fell below what
was needed in both new runs).

**How it is reported.** The failed check is listed under "What failed" in the README and named in the paper's
abstract. The earlier verdicts are not changed after the fact; the README says which earlier pass did not hold up.
The reviewer checked every number against the result files and made me tone down three sentences.

**What's next.** The reviewer's remaining worry: the main Ms. Pac-Man result (older frames remove the parking) comes
from a single training run of that model. It asked for one more training run with a new seed (about $4-5 of the $16
usable), with the pass/fail rule fixed in advance. After that the work stops and I tell you the arXiv files are
ready.

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

