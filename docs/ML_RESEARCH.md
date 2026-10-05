# ML Research Foundation

The Charter's production enemy AI remains the existing heuristic and fixed learned-linear policy system. The `game.ml` package is dev/research infrastructure only; it does not add or promote a neural policy into gameplay.

## Observation contract

Enemy-decision research uses the versioned `enemy-decision-candidate-v1` feature schema. It preserves every feature currently emitted for a legal `EnemyDecisionCandidate` and encodes them in an explicit fixed order. Extraction fails if a required feature disappears or an unexpected feature appears.

Outcome, reward, future-state, seed, policy, actor-instance, target-instance, and encounter identity fields are metadata or labels only. They must never be inserted into candidate observation vectors.

## Dataset unit

One dataset record represents one enemy decision, not one independent candidate row. It contains the complete legal candidate collection emitted by the simulator trace and one explicit chosen-candidate index. Candidate legality, combat resolution, targeting, formation, damage, and stochastic outcomes remain simulator authority.

Dataset extraction starts fresh simulator encounters and records the existing `EnemyDecisionTrace` / `EnemyDecisionRecord` output. It does not recreate combat rules in the research package.

## Held-out split contract

Split membership is assigned by top-level simulation seed and is intentionally easy to inspect:

- train: seeds 1-256
- validation: seeds 10001-10064
- test: seeds 20001-20128

The ranges are disjoint. Every decision produced from one top-level seed stays in the same split, including all legal candidates for each decision. The test split is held out for final evaluation and must not be used for tuning, feature selection, normalization fitting, early stopping, or hyperparameter choice.

Hero-policy and encounter sets are explicit extraction configuration rather than hard-coded dataset semantics. This supports initial collection against `naive`, `damage_race`, `survival`, `anti_mark`, and `conservative`, while leaving later distribution-shift evaluation policies separate.

## Provenance

Research manifests are small deterministic records that can capture the feature/split schema versions, repository commit, content hashes, encounter and hero-policy sets, extraction settings, runtime versions, and later model/normalization/optimizer/checksum metadata.

The current shaped reward remains a baseline label/diagnostic. This foundation does not modify reward logic. Future supervised imitation targets, revised reward definitions, and simulator outcome metrics should be distinguished explicitly in artifact provenance rather than silently changing historical meaning.

## Checkpoint 2 supervised-imitation result

Checkpoint 2 qualified the supervised research pipeline rather than the strategic quality of the teacher. On held-out in-distribution seeds, the same-objective linear model reached about 99.56% heuristic top-1 agreement and the three small MLPs averaged about 99.96%; the nonlinear gain was below the predeclared one-percentage-point materiality threshold. Under the held-out company_survival policy shift, the full MLPs degraded materially while the linear and heuristic-visible models generalized better. Near-perfect in-distribution imitation reproduced the heuristic's simulator behavior, but the heuristic itself was not shown to be a strong enemy policy.

## Long-term optimization principle

Checkpoint 2 uses the heuristic enemy policy only as a supervised teacher to qualify the research pipeline. High imitation accuracy means a model reproduced that ranking relationship; it is not evidence that the heuristic is strategically optimal.

For later combat-success optimization, keep the responsibility split explicit:

- kit and simulator rules define what an enemy can legally do;
- the learned policy should discover what it should do to maximize encounter success;
- balance analysis compares emergent strong play with designer intent.

Signature-ability use, setup/payoff conversion, focus fire, support disruption, resource drain, positioning, target switching, and combo frequency are diagnostic telemetry. They should describe the strategy a strong policy discovered rather than become direct rewards for behaving according to an authored archetype. If a strong unconstrained policy ignores an intended combo, treat that first as balance evidence about the kit rather than a reason to force the combo through reward shaping.

The 39-feature schema remains frozen for Checkpoint 2 comparability. Before any later win-optimization experiment, audit every feature and distinguish neutral state/action information from prescriptive heuristic guidance. Do not automatically carry role/archetype-flavor or other behavior-prescribing engineered signals into an optimizer merely because the imitation teacher used them.

## Checkpoint 3 counterfactual outcome-ranking result

Checkpoint 3 qualified deterministic one-step counterfactual evaluation over real simulator branches. Each sampled source decision is replayed from the same encounter seed and prefix, with state, RNG, and legal-candidate identity verified before forcing exactly one candidate and returning to the fixed heuristic continuation. A control branch must reproduce the original continuation exactly. The CP3 optimizer uses a new 81-feature factual observation schema; the audit classified 64 features as neutral factual and 17 as derived factual, with no prescriptive heuristic-guidance features admitted.

The sampled heuristic distribution contained substantial Pareto outcome signal but almost no victory signal. TRAIN produced 9,493 decisive candidate pairs and VALIDATION 4,892; the heuristic action was Pareto-dominated in about 40% of sampled states, yet neither split contained a heuristic loss-to-win opportunity. On sealed held-out TEST, the validation-selected MLP reached 79.25% pairwise accuracy in distribution, 73.79% under the tactical shift, and 67.05% under company_survival.

One-step policy improvement was real but local: on in-distribution TEST states the learned choice was preferred to the heuristic in 117 states versus 44 favoring the heuristic, with 250 ties and 29 incomparable outcomes. The tactical shift was 21 versus 5, and company_survival was 22 versus 11, but there were no loss-to-win flips and company_survival included one win-to-loss regression. CP3 therefore shows improved party-deterioration outcomes under Q^(heuristic)(s,a); it does not establish improved enemy victory rate.

Recursive deployment remains exploratory rather than qualified because the learned policy visits states outside the heuristic-generated training distribution. The exploratory signal is nevertheless promising for the enemy objective: under company_survival, enemy win rate rose from 3.41% for the heuristic to 27.27% for the learned policy; the in-distribution rollout moved from 0% to 0.23%, while the tactical shift stayed at 0% but reduced final hero HP. Future policy-improvement work must treat one-step counterfactual quality and recursively deployed policy quality as separate questions and gather/evaluate states induced by the improved policy before any gameplay promotion.

## Checkpoint 4 bounded policy-iteration result

Checkpoint 4 performed exactly one policy-iteration step using the frozen CP3 policy as both source and continuation policy. Policy-induced TRAIN and VALIDATION states contained materially richer victory information than CP3: TRAIN had 35 enemy-win branches across 10 win/loss-varying states and VALIDATION had 28 across 5, whereas both CP3 splits had zero. The CP3 policy was also less often locally dominated than the original heuristic (32.4% of CP4 TRAIN states and 28.6% of VALIDATION states, versus 39.7% and 40.7% for the CP3 heuristic-source baselines).

The resulting CP4 policy did not show uniform further improvement. On held-out one-step comparisons it was essentially flat in-distribution (31 decisions preferred versus 33 for CP3), better under tactical (12 versus 8), and better under company_survival (6 versus 2, including one loss-to-win flip). Recursive company_survival evaluation exactly reproduced the CP3 policy's prior 27.27% enemy-win rate, while the CP4 policy reached 25.00%; the paired difference was small and not distinguishable on these 88 cases (exact McNemar p=0.815). Both learned policies remained far stronger than the 3.41% heuristic baseline. This isolates policy-iteration stability and continuation/distribution effects as the next research problem rather than supporting another automatic iteration or gameplay promotion.

Behavioral telemetry remained diagnostic only. The CP4 policy reduced Mark use further while partially restoring Guard/Grab and increasing Bite relative to the CP3 policy; no enemy kits were changed. The reserved future-confirmation seed range 30001-30128 remains untouched.

## Checkpoint 5 untouched-seed confirmation

Checkpoint 5 evaluated the frozen heuristic, CP3 policy, and CP4 policy on the reserved untouched seed range 30001-30128 without retraining or tuning. Under company_survival the heuristic won 60/1408 cases (4.26%), the CP3 policy 330/1408 (23.44%), and the CP4 policy 357/1408 (25.36%). Paired comparisons confirmed both learned policies over the heuristic (+19.18 and +21.09 percentage points respectively), while CP4 versus CP3 was +1.92 points with a 95% paired interval crossing zero and exact McNemar p=0.094; treat the two learned policies as approximately equivalent rather than claiming another policy-iteration gain.

The large learned-policy advantage is real but not universal: enemy wins remain concentrated in company_survival and in a subset of longer/harder encounters. Low Mark, Grab, and Guard usage all replicated on untouched seeds, while direct-pressure selection remained high. Learned policies also showed less target concentration, lowest-HP targeting, repeated focus, and Downed-target attacks than the heuristic in company_survival, so stronger play did not manifest as uniformly harsher focus fire. These results support early policy stabilization, do not justify another identical greedy iteration, and do not warrant production promotion or immediate kit changes. A separate frozen-policy balance-perturbation experiment is the next research direction if this line is continued.
