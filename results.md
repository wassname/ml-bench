<!-- Text before the first heading sits above the chart on the page. Every section below it sits
     under the table, so the chart stays near the top. -->

# wassname-ml-bench

Which machine learning model should you work with? This bench asks fifteen questions from
[wassname](https://wassname.org)'s own research and scores each answer against the one he reached at
the time, so 1.00 is his answer. The work is obscure and some of it is newer than the models, which
means a model has to work the answer out rather than recall it. Code and numbers:
[github.com/wassname/ml-bench](https://github.com/wassname/ml-bench).

## Score against cost

### Table: wassname-ml-bench

15 problems from the research of [wassname](https://wassname.org). The best model scores
0.77, where 1.00 is wassname's own answer.

| model          |   score↑ | +-     | answered   |   answers/q |   $/run↓ |   tok/answer↓ |   pts/Mtok↑ |   AP#1↑ |   CW#2↑ |   KL#3↑ |   MC#4↑ |   NP#5↑ |   OL#6↑ |   PC#7↑ |   SV#8↑ |   TR#9↑ |   TS#10↑ |   VG#11↑ |   VJ#12↑ |   AB#13↑ |   LG#14↑ |   LS#15↑ |
|:---------------|------------------:|:-------|:-----------------------|------------:|---------:|----------------------------:|------------------------:|----------------------------:|---------------------------------:|-----------------------------:|---------------------:|----------------------:|--------------------------:|----------------------:|-----------------------:|-----------------------:|------------------------------:|--------------------------------:|----------------------:|--------------------------:|-------------------------:|-------------------------:|
| claude-fable-5 |             +0.77 | ±0.065 | 15/15                  |           1 |     $1.5 |                       1,653 |                      31 |                       +0.70 |                            +0.02 |                        +0.84 |                +0.82 |                 +1.01 |                     +0.96 |                 +0.79 |                  +0.90 |                  +1.06 |                         +0.63 |                           +0.61 |                 +1.01 |                     +0.82 |                    +0.67 |                    +0.66 |

1.00 is on par with wassname's own answer, and a negative score means the answer hit a trap. Bold is
used to mark the best value and every cell within one error bar of it. A * marks a question the model
refused. Every model answers at reasoning effort 'lowest listed', which the note under the table
qualifies. Eval version 102.
 

## Notes on the main table

### The judges

We use multiple judges. To avoid bias each judge is read through its own two anchors, so a point
means the same on every judge. To avoid self bias we remove all judgement of models from the same
company, including their own. This is fine because the calibration and the number of judges keep the
mean score about the same. <!-- -- wassname -->

Below we show how each judge grades an off topic answer (should be zero) and an ideal answer (should
be one).

| judge                           |   judgments |   off-topic |   gold |    gap |   leniency |
|:--------------------------------|------------:|------------:|-------:|-------:|-----------:|
| openai/gpt-oss-120b             |          15 |      -0.016 | +0.996 | +1.012 |     +0.043 |
| google/gemma-4-31b-it           |          15 |      +0.010 | +0.990 | +0.980 |     +0.016 |
| qwen/qwen3.7-flash              |          15 |      -0.004 | +0.992 | +0.996 |     +0.008 |
| thinkingmachines/inkling-small  |          15 |      -0.020 | +0.988 | +1.008 |     -0.005 |
| deepseek/deepseek-v4-flash-0731 |          14 |      -0.008 | +0.997 | +1.005 |     -0.062 |
