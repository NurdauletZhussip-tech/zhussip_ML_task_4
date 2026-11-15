# Prompts for GitHub Copilot — Task 4 (feed them one at a time, in order)

HOW TO USE
1. Put `copilot-instructions.md` into the repo as `.github/copilot-instructions.md`
   (Copilot reads it automatically in every chat). Also attach it with `#file` in the first prompt to be sure.
2. Use Copilot Chat in **Agent mode** so it can edit the notebook and run git commands.
3. Send ONE prompt at a time. Read the output, check the numbers make sense, then send the next.
4. Wherever you see `[PASTE ...]`, paste the exact text of that Part from the handout in Canvas (Week 08).
   The grading table alone does not contain the exact rules for `oob_mask`, `hard_vote`, `adaboost_step`,
   so Copilot must get them from the handout.
5. Replace `lastname` with your surname everywhere.

---

## P0 — Setup, first commit

```
Read #file:.github/copilot-instructions.md and follow it for this whole session.
Context: Task 4, Random Forests & Ensembles, data in data/spambase.csv.

Do this:
1. Check `git status`, `git remote -v`, and that .gitignore ignores the virtual environment folder
   .lastname_ML_task_4 and .ipynb_checkpoints. Do not change requirements.txt.
2. Create lastname_task4.ipynb in the repo root with:
   - a title markdown cell (Task 4, my name, course MOP 3231)
   - a setup code cell: imports, SEED = 3231, print of python/numpy/pandas/sklearn/matplotlib versions
   - load data/spambase.csv, show shape and class balance
   - ONE stratified train/test split [PASTE test fraction from handout, otherwise 0.25], saved in
     X_train, X_test, y_train, y_test; print their sizes and spam share
   - ONE shared CV object `cv` [PASTE number of folds from handout, otherwise StratifiedKFold(5, shuffle=True, random_state=SEED)]
3. Run the notebook top to bottom, then commit "part0: setup, split, shared CV" and push to origin main.
```

---

## P1 — Part 1: Bagging & OOB

```
Part 1 of the handout (Bagging & OOB) is below. Implement it in the notebook under a "## Part 1" header.

[PASTE FULL TEXT OF PART 1 FROM THE HANDOUT, including the rules for oob_mask]

Required results (each as code + printed output + a short interpretation cell quoting the numbers):
a) `oob_mask` exactly as the rules say, with a check on a tiny example.
b) Mean out-of-bag share per tree compared with (1 - 1/n)^n, n = training size.
c) My own out-of-bag accuracy computed with oob_mask equals the forest's `oob_score_`
   (print both and whether they are equal; if not exactly equal, explain the reason from the handout rules).
d) CV table for the FIVE models from the handout, on the SAME shared `cv` folds: mean ± std of accuracy.
e) Does bagging help each of the two base models (decision tree and logistic regression)?
   Support with the PAIRED per-fold differences (mean difference, std of differences). If the difference is
   inside the fold spread, say there is no clear gain.
f) The forest's oob_score_ vs its CV mean vs its test accuracy with standard error sqrt(p(1-p)/n_test).

Use the existing X_train, y_train, X_test, y_test, cv, SEED. Run, then commit
"part1: bagging, oob_mask, CV table, oob vs cv vs test" and push.
```

---

## P2 — Part 2: Evaluation (ROC/PR, 1 % FPR threshold, importances)

```
Part 2 of the handout (Evaluation) is below. Implement it under "## Part 2".

[PASTE FULL TEXT OF PART 2 FROM THE HANDOUT]

Required:
a) Fit the single decision tree and the random forest on the training split. On the TEST split draw
   ROC curves and precision-recall curves for both (labelled, titled, with legend). Print AUC and average
   precision for both.
b) Choose a decision threshold for the forest so that false-positive rate <= 1 %, using ONLY out-of-fold
   predicted probabilities of the TRAIN split (cross_val_predict with method="predict_proba" and the shared cv).
   Apply that threshold ONCE to the test split. Report confusion matrix, precision, recall and FPR on test,
   and state clearly whether the 1 % target held on test (mention that with ~1000 test rows, ~600 legitimate,
   FPR has a large sampling error; give a binomial standard error or interval).
c) Permutation importance and impurity importance of the forest, top five features each, in one table.
   Explain ONE difference between the two rankings (e.g. impurity bias toward features with many split points,
   correlated features sharing importance, permutation measured on held-out data).

Never use the test split to choose the threshold. Run, then commit
"part2: roc/pr, 1pct fpr threshold from oof probs, importances" and push.
```

---

## P3 — Part 3: Voting

```
Part 3 of the handout (Voting) is below. Implement it under "## Part 3".

[PASTE FULL TEXT OF PART 3 FROM THE HANDOUT, including the rules for hard_vote]

Required:
a) `hard_vote` exactly as the rules say (own implementation, do not call VotingClassifier inside it).
b) Show that hard_vote reproduces sklearn VotingClassifier(voting="hard") on the TEST split
   (np.array_equal, print True/False).
c) Pairwise disagreement table between the three members on the test split.
d) CV table on the SAME shared folds for: the three members, hard voting, soft voting (mean ± std).
e) Conclusion versus the BEST single member: report the paired fold differences; if a difference is inside
   the fold spread, declare no winner.

Run, then commit "part3: hard_vote, disagreement table, voting CV" and push.
```

---

## P4 — Part 4: Stacking

```
Part 4 of the handout (Stacking) is below. Implement it under "## Part 4".

[PASTE FULL TEXT OF PART 4 FROM THE HANDOUT]

Required:
a) My own out-of-fold stacking (base-model predictions from cross_val_predict on the train split, meta-learner
   fitted on them, base models refitted on full train for test prediction). It must reproduce
   StackingClassifier: compare meta-learner weights (np.allclose) and test accuracy; print both.
b) Meta-learner weights trained on IN-SAMPLE base predictions vs on OUT-OF-FOLD base predictions, in one table.
   Explain why the tree's weight changes (in-sample the unpruned tree predicts the training labels almost
   perfectly, so the meta-learner over-trusts it; out-of-fold its real, weaker quality shows).
c) CV comparison on the same folds: stacking, soft voting, best single member. Paired differences.
   Declare no winner if the difference is inside the fold spread.

Run, then commit "part4: own oof stacking, weights in-sample vs oof, CV comparison" and push.
```

---

## P5 — Part 5: AdaBoost & label noise

```
Part 5 of the handout (AdaBoost & noise) is below. Implement it under "## Part 5". This is the biggest part (6 points),
do it in two steps and run after each.

[PASTE FULL TEXT OF PART 5 FROM THE HANDOUT, including the rules for adaboost_step and the lecture's 3-round example]

Step A:
a) `adaboost_step` exactly as the rules say.
b) Reproduce the lecture's three-round example (print my numbers next to the lecture numbers).
c) Reproduce scikit-learn AdaBoostClassifier `estimator_weights_` for the FIRST FIVE rounds
   (same base stump, same SAMME algorithm as in the handout; np.allclose, print max difference).
d) Plot training error and test error by boosting round (staged_predict), labelled. Choose the number of
   rounds using CV on the TRAINING folds only (not the test split), mark the chosen round on the plot,
   and compare with where the test curve is actually best.

Step B (label noise):
e) Flip 20 % of the TRAIN labels (five different seeded draws derived from SEED; also 0 % noise). Test labels stay clean.
   Table for decision tree, random forest, AdaBoost at 0 % and 20 %: mean ± std test accuracy over the five draws,
   and for AdaBoost the share of its final sample weight sitting on the flipped rows (use my own
   adaboost_step loop or the weights it produces, as the handout says).
f) Conclusion: which model loses most accuracy from label noise, using the spread over draws, and explain the
   mechanism (boosting keeps increasing the weight of rows it cannot fit, i.e. the mislabelled ones; bagging averages
   them out).

Commit after Step A: "part5a: adaboost_step, reproduction, error by round, rounds from CV".
Commit after Step B: "part5b: label noise table, weight share on flipped rows". Push both.
```

---

## P6 — Part 6: Conclusion and reproducibility

```
Do Part 6 under "## Part 6 — Conclusion".

[PASTE FULL TEXT OF PART 6 FROM THE HANDOUT]

1. Write a recommendation for the spam filter in AT MOST SIX sentences. It must contain: the chosen model,
   the threshold (from Part 2), the numbers it rests on (precision, recall, FPR on test, CV mean ± std),
   and what is inside the noise (differences smaller than the fold spread). Use ONLY numbers that appear in
   earlier cells of this notebook; print a small summary table with them right above the text.
2. Check reproducibility: versions printed in the first cell, SEED stated, no unseeded randomness
   (grep the notebook for random calls without random_state).
3. Restart the kernel and Run All; fix any error. Save the notebook WITH outputs.
4. Verify with `git ls-files` that lastname_task4.ipynb, requirements.txt, data/spambase.csv and README.md
   are tracked, and that `git log --oneline` shows at least three commits.
5. Commit "part6: conclusion, executed notebook, final check", push to origin main, and print the final
   commit URL https://github.com/<user>/<repo>/commit/<full-hash>.
```

---

## P7 — Final audit (optional, send last)

```
Audit lastname_task4.ipynb against this grading table and list, per criterion, what is present, what is missing,
and the cell where it is. Do not change anything yet, only report.

1 Bagging & OOB (5): oob_mask per rules; oob share vs (1-1/n)^n and own OOB accuracy == oob_score_;
  CV table for five models on same folds; does bagging help tree and logistic regression, with paired fold diffs;
  forest oob_score_ vs CV mean vs test accuracy and SE.
2 Evaluation (4): ROC+PR of tree and forest on test with AUC and AP; threshold for FPR <= 1 % from out-of-fold
  train probabilities applied once to test with confusion matrix, precision, recall, FPR and whether target held;
  permutation vs impurity importances top five with one difference explained.
3 Voting (4): hard_vote per rules; reproduces VotingClassifier hard on test + pairwise disagreement table;
  CV table for 3 members + hard + soft with conclusion that declares no winner inside fold spread.
4 Stacking (4): own OOF stacking reproduces StackingClassifier (weights, test accuracy); weights in-sample vs OOF
  and tree-weight change explained; CV comparison with no winner inside fold spread.
5 AdaBoost & noise (6): adaboost_step per rules; reproduces lecture example and estimator_weights_ for 5 rounds;
  labelled error-by-round plot with rounds chosen on train folds and compared with test curve; noise table
  (tree, forest, AdaBoost, 0 % and 20 %, five draws, weight share on flipped rows); conclusion using spread
  and mechanism.
6 Conclusion & reproducibility (2): <= 6-sentence recommendation with threshold and numbers; versions, SEED,
  runs top to bottom, executed notebook + requirements.txt + data in repo, >= 3 commits.

After the report, wait for my confirmation before fixing anything.
```
