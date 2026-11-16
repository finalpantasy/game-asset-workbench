# Maintaining the gaw skill

**The rule: a change to a command's behavior, flags or output updates the matching skill reference in the same change.** A new command adds a reference entry and a router line in `SKILL.md`. `cargo test -p zidane-inspector --test skill_coverage` fails when a command in `gaw help` is missing from the skill. It checks only that each command is mentioned; whether the text is still true is on whoever changed the command.

The rest of this file is for the person or agent who keeps the skill current. Users of the skill do not need it.

The canonical copy lives next to the source at `workbench/skills/gaw/`. The public repo gets a copy at push time, with `local/` left out.

## Where feedback comes from

Agents append one line each to `local/feedback.md` (not committed): `date | command | what happened | the fix`. That file is the queue. Nothing else feeds it.

## Review pass

Run a pass after a batch of sessions or when the file passes about 20 lines.

1. Read `local/feedback.md` top to bottom.
2. Check each line against `gaw help` and the current code. Lines about flags that changed are stale: delete them.
3. Sort each remaining line into one of three bins.

| The line says | Do this |
|---|---|
| An agent chose the wrong command or order, or lacked a fact | Fold it into `SKILL.md` (router, loops, never-list) or the matching `references/*.md` |
| A command did the wrong thing, crashed, gave a misleading pass, or needs a workaround every time | It is a tool bug. Fix it in code, add a test, then drop the workaround line |
| Only this PC (paths, runtime, process rules) | Put it in `local/notes.md` |

4. Remove folded lines from `feedback.md` or mark them `done <date>`. Keep unresolved ones.

## Rules for edits

- Flags live in `gaw help`, not here. If an edit would copy a flag list into the skill, point at `gaw help` instead.
- Keep `SKILL.md` a router: when to use which command, the loops, the never-list. Per-command detail goes in `references/`. Game-specific facts go in `references/ff9.md` (or a new `references/<game>.md`).
- Add a never-rule only for a mistake that happened and cost real work. State what to do instead.
- A fact needs a source: a README line, a check output, a test. Drop guesses.
- Keep the public files free of the maintainer's name, email and absolute paths. Private paths belong in `local/`.

## When a feedback line means a code fix

Fix the tool, not the skill, when any of these holds:

- The same line appears twice from different sessions.
- The workaround is a flag combination every user must remember.
- A check passed while the result was visibly wrong (a check that cannot catch the problem).
- An error message does not say what to do.

After the fix, delete the workaround from the skill and keep one line in the reference only if the old behavior can still bite (an older build).

## Test the skill

After a substantial edit, give a fresh agent only the skill and a realistic task (for example: check a candidate GLB against stock, say what changed and whether the blink boxes survived). Note every place it hesitated, guessed a flag, or read the wrong reference, and fix those places. One round is usually enough.

## Scans before commit

Run the banned-phrase and structure scans on `SKILL.md`, `MAINTAINING.md` and `references/` (docs genre), and grep the public part for personal names, emails and user paths. `local/` stays out of git: it is in `.gitignore`.
