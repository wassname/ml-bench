<!-- Prose rewritten by Pi (annoy-less, rewrite mode); wassname, edit at will. -->

# wassname-ml-bench

Which machine learning model should you work with? Public leaderboards test models on questions
written by someone else. This bench tests them on twelve questions from
[wassname](https://wassname.org)'s own research. Each question has the answer wassname reached at
the time.

His research is obscure, and some of it came after these models were trained. The answers are not
in the training data, so a model must work them out. A score of 1.00 means the model reached
wassname's answer. A higher score means the judge rated the model's answer above his. Wassname
then writes a better reference answer for that question.

The questions stay private. A public question ends up in the next model's training data, and the
bench would then test recall. The page shows each model's score, the domain of each question, and
the cost of the run.

## Limitation: every model answers at low reasoning effort

Most benchmarks run models at their highest reasoning setting, which removes thinking budget as a
variable and shows each model at its best. This bench runs them at the lowest setting, because it
pays for its own tokens. So these scores are not comparable with a public leaderboard, and a model
that leans on long deliberation is under-served here.

The setting is also a weaker control than it looks. At one effort word the models in this table
write between 10 thousand and 182 thousand tokens for the same twelve questions, a factor of 17, so
the same label buys very different amounts of thinking from different vendors.

One model has been measured both ways. grok-4.6 scores 0.06 higher at high effort than at minimal,
consistent across three runs, for about six times the tokens. A row whose name carries
`(effort:high)` was run that way.
