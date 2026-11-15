# Copilot instructions — Task 4: Random Forests & Ensembles (MOP 3231)

You are helping a student build ONE notebook, `lastname_task4.ipynb`, for a graded Machine Learning task
(25 points, spam classification on `data/spambase.csv`). Follow these rules in every reply.

## 0. Hard facts about the project

- Repo root contains: `data/spambase.csv`, `requirements.txt`, `make_dataset.py`, `README.md`, `.gitignore`,
  the notebook `lastname_task4.ipynb`, and the virtual environment folder `.lastname_ML_task_4/`.
- Data: 4,204 rows, 57 numeric features, binary label `spam` (1 = spam, 39.9 %). Already de-duplicated.
  Do NOT scale, clip, round or otherwise alter the raw values (trees do not need scaling; scale only inside a
  `Pipeline` for logistic regression).
- Libraries are pinned in `requirements.txt`. NEVER edit, upgrade or add to `requirements.txt`.
  Use only numpy, pandas, scikit-learn, matplotlib (and anything already in `requirements.txt`).
- Work in ONE notebook. Do not create extra .py modules for the solution; helper functions live in notebook cells.
- Never commit the virtual environment, `.ipynb_checkpoints`, or large generated files.

## 1. Reproducibility rules (graded)

- First code cell: imports, `SEED = 3231` (use the seed the handout states if it differs), and a printout of
  Python, numpy, pandas, scikit-learn and matplotlib versions.
- Pass `random_state=SEED` (or a `np.random.default_rng(SEED)`-derived value) to EVERY random component:
  split, CV, trees, forests, AdaBoost, bootstrap, noise draws. No unseeded randomness anywhere.
- ONE stratified train/test split, made once at the top, with the test fraction from the handout.
  The test split is touched only for the final numbers of each part. Never tune, choose thresholds, choose
  hyperparameters or choose the number of boosting rounds on the test split.
- ONE shared CV object (e.g. `StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)`, or what the
  handout says), built once and reused for every comparison, so differences are PAIRED fold by fold.
- The notebook must run top to bottom in a fresh kernel with "Restart & Run All" and no errors.

## 2. Statistical-honesty rules (graded)

- Report every comparison as mean ± std over folds AND the paired per-fold differences.
- If a difference is smaller than the fold spread (std of the paired differences), the conclusion must say
  "no winner / inside the noise". Never declare a winner on a difference inside the spread.
- Test accuracy is reported with its standard error, sqrt(p(1-p)/n_test).
- Do not write a conclusion before the numbers exist. Conclusions must quote the actual numbers printed
  by the cell above them. If a number contradicts the expected textbook result, say so and explain; do not
  adjust code until the "expected" result appears.
- Every plot: title, axis labels, legend, readable fonts.

## 3. Implementing "own" functions (graded: they must follow the handout rules)

The handout defines exact rules (signature, inputs, outputs, allowed calls) for `oob_mask` (Part 1),
`hard_vote` (Part 3) and `adaboost_step` (Part 5). Before writing any of them:
1. Re-read the handout text the student pasted for that part.
2. Implement exactly that signature and rules; do not call the scikit-learn class you are supposed to
   reproduce inside your own function.
3. Add a verification cell that compares your result with the scikit-learn result using
   `np.array_equal` / `np.allclose` and prints `True`/`False` plus the max absolute difference.
If the handout text for a part has not been pasted into the chat, STOP and ask for it. Do not guess the rules.

## 4. Per-part checklist (what the grader looks for)

Part 1 — Bagging & OOB: `oob_mask` per rules; mean out-of-bag share per tree vs (1 − 1/n)^n; own OOB accuracy
equals `oob_score_`; CV table for the five models on the same folds; does bagging help the tree and logistic
regression, backed by paired fold differences; forest `oob_score_` vs its CV mean vs test accuracy ± SE.

Part 2 — Evaluation: ROC and PR curves of tree and forest on the test split with AUC and average precision;
threshold chosen on OUT-OF-FOLD probabilities of the TRAIN split (`cross_val_predict(..., method="predict_proba")`)
for FPR ≤ 1 %, applied ONCE to test; confusion matrix, precision, recall, FPR; state whether the 1 % target held;
permutation vs impurity importances, top five each, explain one difference.

Part 3 — Voting: `hard_vote`; reproduces `VotingClassifier(voting="hard")` on test; pairwise disagreement table;
CV table for three members + hard + soft voting; conclusion vs best member that respects the fold spread.

Part 4 — Stacking: own out-of-fold stacking reproduces `StackingClassifier` (meta-learner weights and test
accuracy); meta-learner weights with in-sample vs out-of-fold predictions and why the tree's weight changes
(in-sample tree predictions are overfit, nearly perfect, so the meta-learner over-trusts them); CV comparison of
stacking, soft voting, best member; conclusion respects the fold spread.

Part 5 — AdaBoost & noise: `adaboost_step`; reproduces the lecture's three-round example and
`estimator_weights_` for the first five rounds; labelled plot of train and test error by round; number of rounds
chosen on TRAIN folds only, then compared with the test curve; label-noise table (tree, forest, AdaBoost) at 0 %
and 20 % noise over five draws, with the share of AdaBoost's weight on the flipped rows; conclusion on who loses
most, using the spread over the draws and explaining the mechanism (boosting up-weights hard/mislabelled rows).
Flip labels only in the TRAIN data; evaluate on the clean test labels unless the handout says otherwise.

Part 6 — Conclusion: recommendation for the spam filter in AT MOST six sentences, with the chosen threshold,
the numbers it rests on, and what is inside the noise.

## 5. Notebook style

- Structure: one `##` markdown header per Part, a short markdown cell stating what is being done and why,
  then code, then a markdown cell with the interpretation (numbers quoted from the output).
- Keep code simple and readable; short comments. The student must be able to explain every line.
- Print compact tables with pandas (`round(4)`). No wall-of-text outputs, no huge debug prints.
- Language of the notebook text: English.

## 6. Git workflow (you may run terminal commands)

- Work on branch `main`; remote `origin` is the student's OWN repo. Check with `git remote -v` first.
- Commit after EACH finished part, never one giant commit. Minimum 3 commits is required; aim for 8–9.
- Commit message format: `partN: short description` (e.g. `part1: bagging, oob_mask, CV table`).
- Before each commit: `git status`, make sure only intended files are staged
  (notebook, and at the end data + requirements). Never `git add .` if the venv is not ignored — check
  `.gitignore` first. Never `git add -A` blindly.
- After each commit: `git push origin main`. Never use `--force`.
- If a push fails, show the error and stop; do not try workarounds that rewrite history.
- Before the FINAL commit: Restart & Run All so the notebook is saved WITH outputs
  (or `jupyter nbconvert --to notebook --execute --inplace lastname_task4.ipynb`), then verify that
  `lastname_task4.ipynb`, `requirements.txt` and `data/spambase.csv` are tracked
  (`git ls-files`), push, and print the final commit URL
  `https://github.com/<user>/<repo>/commit/<hash>` for submission in Canvas.

## 7. When in doubt

Ask one short question instead of guessing. Never invent numbers. Never hardcode results in markdown that are
not produced by code in the notebook.
