# Scenario suite changelog

The suite is versioned separately from the benchmark and the results schema, because
a scenario change alters what a score means. `compare.py` warns when result files were
produced by different suite versions.

`suite_version` is the human-facing number. `suite_fingerprint` in each results file is
a content hash and changes on any edit, including ones too small to warrant a version
bump.

## 1.1.0

Fixed `sr-02-timetable-tomorrow`, which could pass on the wrong day's data.

7A is taught the same three subjects on Monday and Tuesday, so the subject-name
assertions could not distinguish them. A model that called `get_timetable` with
`day: "today"` received Monday's timetable and still scored 1.0 on
`final_answer_factual`. The scenario as a whole still failed, because
`correct_tool_arguments` caught the wrong day, so the verdict was right for the wrong
reason - but the factual dimension was measuring nothing.

The answer must now contain `Tuesday` and must not contain `Monday`. That is the one
fact that distinguishes the two tool results, since `get_timetable` returns the
resolved day. Ground truth now records both days so the discriminator is asserted
against the simulator rather than assumed.

Scores for this scenario are not comparable across 1.0.0 and 1.1.0.

Known limit of this fix. `final_answer_factual` is the fraction of checks passed, so a
wrong-day answer still earns partial credit: it fails `contains Tuesday` but passes the
three subject checks, because 7A's Monday and Tuesday share a subject set, and it
passes `excludes Monday` whenever the model writes "tomorrow" in prose rather than
echoing the day the tool returned. Measured on the wrong-day answer: 1.00 under 1.0.0,
0.833 under 1.1.0. Full factual credit now genuinely requires Tuesday, and the scenario
itself fails (0.578, below the 0.80 pass bar), which is the level the pass rule acts on.

Closing the remaining gap needs an ordered check, because period order is the only
other thing that separates the two days: Tuesday runs Urdu, English, Science, English,
Urdu and Monday runs English, Science, Science, Urdu, English. Substring checks are
unordered, so this would mean a new `answer_must_contain_in_order` primitive in the
scorer. Not added here: it is a new scoring primitive rather than a scenario fix, and
it would invalidate the four model runs recorded against 1.1.0.

## 1.0.0

Initial 54-scenario suite across ten categories.
