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
