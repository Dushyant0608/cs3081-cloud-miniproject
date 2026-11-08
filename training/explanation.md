# Task 2: Distillation method explanation

## Sequence-level vs. logit-based distillation

**Sequence-level distillation** (what this project uses) trains the student on the
teacher's final generated *output* — the actual text the teacher produced for each
command. The student never sees anything about how the teacher arrived at that output;
it only sees pairs of (command, teacher's text reply) and learns to reproduce the reply
given the command, exactly like ordinary supervised fine-tuning on a dataset someone
else labeled. The teacher is only needed once, to generate that dataset; after that the
student is trained independently of the teacher.

**Logit-based distillation** trains the student to match the teacher's full output
*probability distribution* at each generation step (its logits over the whole
vocabulary), not just the single token the teacher happened to pick. This gives the
student a much richer training signal per example — e.g. which other tokens the teacher
considered plausible and by how much — and is what the Hugging Face TRL
`DistillationTrainer` referenced in the handout implements.

## Why logit-based distillation was not possible here

Logit-based distillation requires access to the teacher's logits (or at least its
output probability distribution) for every generated token, computed with the *student's
own tokenizer vocabulary* so the two distributions are directly comparable. In this
project the teacher (phi3.5) is served through Ollama's HTTP API, which returns only
the final generated text, not per-token logits. Even if Ollama exposed logits, phi3.5
and pythia-70m use different tokenizers with different vocabularies, so the teacher's
logits would be over a different token space than the student's and could not be used
directly as a training target without a vocabulary-alignment step the handout explicitly
says is out of scope ("you do not need the teacher's logits"). Sequence-level
distillation sidesteps this entirely because it only needs the teacher's final text.

## One limitation of this approach

Because the student only ever sees the teacher's single chosen output per command, it
has no signal about *how confident* the teacher was or what else was plausible — two
commands the teacher was unsure about and got wrong look identical, in training, to two
commands it was completely confident and correct on. This makes the student more prone
to confidently reproducing the teacher's mistakes (and its own generalization errors) on
unseen validation commands, with no way to distinguish "the teacher was shaky here" from
"this is a clean, well-covered pattern." It also means the student can only ever be as
good, at best, as the subset of teacher behavior that happens to be representable by a
70M-parameter model trained on 1,500 examples — a much weaker model than the 3.8B-param
teacher, and with a tiny fraction of the training data.