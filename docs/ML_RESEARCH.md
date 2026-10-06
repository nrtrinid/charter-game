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


## Checkpoint 6 stateful route-evaluation result

Checkpoint 6 replaced the planned balance-perturbation experiment with a production-backed stateful route evaluator. Research routes now chain `create_encounter_combat` -> frozen-policy combat -> `sync_company_from_combat(..., party_formation)`, so HP, Effort, Downed state, Mortal Wounds, morale, strain/marks, quirks, gear-derived stats, surviving persistent tags, and interactive formation carry between encounters under the real transition contract; Mark is explicitly cleared between combats and no town recovery is inserted. Low-lethality fights such as Probe, Glass Pack, and Shallow Cave survive as meaningful attrition because they consistently remove HP/Effort before later fights despite near-zero isolated wipe rates.

Route context materially changes frozen-policy behavior and lethality rather than simply making every later fight harder. Pattern Cell and Cave Mini Boss become substantially more lethal for learned policies after conditioning, while CP3 pi1 Hunt becomes dramatically less lethal after Probe even though the party enters damaged; CP4 pi2 Hunt remains strong. Across the primary company_survival route suite, true enemy victories were 6/384 for the heuristic, 150/384 for pi1, and 210/384 for pi2, while the broader route-completion endpoint is mixed because heuristic runs frequently hit the historical 20-round research withdrawal cap. Tactical and anti_mark routes reveal learned-policy route failures/victories that isolated CP5 rarely exposed.

The persistence architecture is qualified, but the current completion endpoint is not yet a fully faithful manual-play route environment: interactive combat has no automatic 20-round withdrawal, and several conditioned panels never reach their target because Pattern Cell terminates at that cap. Missing-target contexts must also be treated as unavailable rather than zero behavior. Treat single-combat training as partially misaligned with the expedition objective, do not redesign encounters for low standalone win rate, and modify the paused balance-perturbation plan before any CP7. The route evaluator is promising infrastructure for future Maze intelligence but requires this withdrawal/missingness hardening first.

## Checkpoint 7 production-faithful route-termination result

Checkpoint 7 corrected the research-only termination abstraction without changing production combat. Manual combat has no automatic round-20 withdrawal; the old research runner's 20-round guard was therefore reclassified as an execution-safety concern rather than a fictional outcome. The corrected evaluator runs the frozen policies until production-valid victory/defeat or an explicit censored research stop, with a fixed 2048-transition / 256-round safety budget and conservative cycle semantics. Nine previously consumed short combats reproduced final state, RNG, action/event sequences, and persisted company state exactly.

The correction materially changed the route-strength conclusion. Every former CP6 round-20 withdrawal resolved as ENEMY_VICTORY: 259/259 for pi0 across all CP6 hero-policy routes (258/258 in company_survival), 73/73 for pi1, and 85/85 for pi2 (84/84 company_survival). No primary route remained cycle/safety censored or invalid. Corrected company_survival route enemy-win rates are pi0 264/384 (68.75%), pi1 223/384 (58.07%), and pi2 294/384 (76.56%). Paired evidence makes pi1 significantly weaker than pi0 on this route suite, while pi2 is significantly stronger than both; prior blanket learned-policy superiority was therefore materially contaminated by the old cutoff. Former withdrawals are best described as stalling-but-eventually-terminal, with pi0 especially slow (median eight additional rounds and high no-progress-round frequency).

Fresh-versus-conditioned analysis now treats TARGET_NOT_REACHED as missing rather than zero. Hunt Guarded is consequently INSUFFICIENT_OBSERVATION for conditioned route-role classification rather than NEGLIGIBLE, despite high fresh lethality. Route conditioning still materially changes some policy/encounter outcomes, and Probe/Glass Pack/Shallow Cave remain meaningful attrition contributors. With zero primary censoring/invalid runs, the corrected route evaluator is qualified as a future optimization environment and the paused frozen-policy causal balance perturbation may proceed as a later CP8; CP7 itself makes no production combat or balance change.

## Checkpoint 8 route-aware causal mechanic perturbation result

Checkpoint 8 froze seven modest research-only mechanic perturbations and evaluated them on fresh seeds 50001-50064 using the exact production-faithful CP7 six-route suite and frozen pi0/pi1/pi2 policies. Baseline equivalence matched CP7 exactly across 54 previously consumed cases before new-seed access. Every perturbation ran in an isolated worker process; authored YAML, production combat defaults, production AI, and route generation were unchanged. The frozen protocol checksum is `03cbb9debf4d94843ec87550d712e3d600d7722b45829c35d96d3712c72c2785`.

Under the primary pi2 route endpoint, only the immediate-value Grab perturbation (G1: +1 base damage to Drag Forward/Stalker Hook) produced a clear improvement: enemy route victories rose from 316/384 (82.29%) to 332/384 (86.46%), a paired +4.17 percentage-point effect with 95% CI +2.17 to +6.17 points and exact McNemar p=3.05e-5. The same condition improved matched local-combat enemy victory by +6.25 points and did not trigger the predeclared strategy-collapse diagnostic. The positional-payoff Grab variant (G2: longer drag-applied Mark) was prospectively unobservable to the frozen learned feature schema and produced no route effect, so Grab remains `MIXED_INCONCLUSIVE` rather than a production buff recommendation.

Mark payoff/economics and Guard mitigation/economics did not materially improve pi2 route outcomes, yielding `HEURISTIC_OVERVALUATION` classifications for Mark and Guard under the frozen CP8 rules. Effort Drain economics also did not improve route outcomes; pi2 already selected Effort Drain on every legal opportunity and the free-cost variant triggered `POTENTIAL_STRATEGY_COLLAPSE`, so Effort Drain remains `MIXED_INCONCLUSIVE`. The Guard resolution recorder did not observe consumption events because the production action module holds the damage function by imported reference; the D1 mechanics override itself is independently unit-qualified and route endpoints remain authoritative, but Guard consumption totals should not be used as a diagnostic from this checkpoint.

CP8 found no qualified local-vs-route tradeoff and no evidence that single-combat analysis understated a setup/control mechanic in the tested matrix. Direct-pressure dominance was classified as primarily `RAW_DAMAGE_VALUE`. No production balance change is justified from CP8 sensitivity analysis, even for G1. Pi2 remained the strongest frozen route-level balance oracle across the tested environments. The recommended next broad direction is `ROUTE_AWARE_POLICY_LEARNING`; the evidence also strengthens the case for eventual adversarial Maze-director research, but neither is implemented here.

## Checkpoint 9 route-aware policy-learning result

Checkpoint 9 trained exactly one route-aware tactical policy generation from frozen pi2 source trajectories and pi2 continuation, using production-faithful CP7 route termination and baseline production mechanics. The primary tactical schema remained factual and added only nine visible persistent-party aggregates to the prior 81-feature candidate observation; a separate diagnostic oracle added only encounter index, encounters remaining, and normalized route progress. TRAIN and VALIDATION contained abundant direct route-victory supervision (286 and 127 win/loss decisive pairs respectively), and held-out data confirmed genuine horizon disagreement: 21 of 209 pairs decisive at both horizons reversed local-versus-route preference, with another 44 route-only decisive pairs.

The selected `cp9_route_mlp_seed_20261010` model achieved 72.73% held-out company_survival pairwise route-ranking accuracy, materially above reconstructable pi2 source-choice ranking (55.38%), but recursive deployment failed catastrophically: pi2 won 81/96 held-out company_survival routes (84.38%) while pi3-route won 32/96 (33.33%), a paired -51.04 percentage-point effect with 95% CI -61.09 to -40.99 points and exact McNemar p=3.55e-15. Pi3-route also regressed on isolated combats (27.27% vs pi2 44.32%). Treat this as a continuation/state-distribution and underconstrained-tie failure: one-step pi3 disagreed with pi2 on 63/72 sampled states, but 54 disagreements were route-outcome ties under pi2 continuation.

Representation/model capacity is not the leading explanation. Linear and MLP both reached 72.73% primary TEST accuracy, and the minimal route-context oracle improved only to 74.31% (+1.58 points, below the predeclared 5-point material gap). The route problem is partially observable, but simply leaking horizon into tactical AI is not supported. Behavior shifted sharply under the weak pi3 policy (much more Mark/lowest-HP targeting, much less Grab/Bite); Effort Drain remained selected on 100% of matched held-out legal-opportunity states (8/8 for both pi2 and pi3), so route-aware training did not make that choice more state-contingent. No pi3 promotion, untouched confirmation, production AI/balance change, or immediate pi4 iteration is warranted from CP9. The reserved 63001-63064 confirmation block remains untouched.


## Checkpoint 10 conservative route-policy repair result

Checkpoint 10 preserved CP9, reproduced its failed pi3 deployment, and diagnosed the recursive collapse before constructing one bounded repair. CP9's route ranker was not primarily failing because its top action was usually wrong on sampled source states: its frozen top-1 choice was in the true preferred set on 66/72 sampled held-out states (91.67%). Instead, 91.1% of pi3-versus-pi2 disagreements occurred in EDGE/POOR support relative to the original CP9 TRAIN distribution, party-state divergence grew sharply after repeated deviations, and a frozen continuation audit found 8 reversals among 66 ordered comparable candidate pairs. In a 12-case bounded interaction audit, pi2 won 12/12, one pi3 deviation preserved 7/12, two preserved 7/12, three preserved 6/12, and full pi3 won 0/12. Treat CP9 as primarily a recursive distribution-shift/compounding failure with a material continuation component, not a simple model-capacity or pairwise-ranking failure.

The single CP10 repair retained pi2 as an anchor, added a bounded stable-hash sample of pi3-induced TRAIN states labeled by forcing each legal action and returning to frozen pi2 continuation, trained the same modest linear/MLP scorer family, and selected a conservative deployment gate using VALIDATION only. The frozen repair uses `cp10_route_mlp_seed_20261013` with a `MARGIN_GATED` override threshold of 3.4007568359375. On sealed seeds 72001-72016, pi4-conservative won 80/96 company_survival routes (83.33%) versus pi2 74/96 (77.08%) and failed pi3 32/96 (33.33%). The pi4-minus-pi2 paired difference was +6.25 percentage points with 95% paired CI +0.58 to +11.92 points but exact McNemar p=0.0703; under the frozen rule this is `PRESERVED_BASELINE_WITH_USEFUL_ROUTE_DEVIATIONS`, not `CLEAR_IMPROVEMENT`. Pi4 overrode pi2 on 27.0% of recursive decisions and reached 91.67% held-out top-1 preferred-set accuracy while restoring much of the behavior lost by pi3: Mark fell from 233 uses to 82, Bite recovered from 0 to 83, and Grab rose from 3 to 218 without implementing CP8 G1. Effort Drain remained selected on 83/84 legal opportunities (98.8%), so its strategy-collapse risk persists.

CP10 did not open the reserved 63001-63064 confirmation range and did not promote pi4 or modify production gameplay. The next research step, if authorized, is an untouched pi4 confirmation on that reserved range. One closeout caveat is purely provenance bookkeeping: CP10's sealed `provenance.json` retains stale lifecycle booleans for normalization/training/model-selection despite the checksum-verified trained-model and selection artifacts; do not rewrite the sealed file post-TEST, and treat the concrete artifacts/checksums as authority for those completed stages.

## Checkpoint 11 untouched conservative route-policy confirmation

Checkpoint 11 independently evaluated the exact frozen CP10 `pi4_conservative` policy against the frozen pi2 anchor on the reserved, previously unused seed block 63001-63064. Before seed access, CP10 was committed, the obsolete CP9 pre-run artifact-root test was repaired in a separate scientific-behavior-neutral maintenance commit, the full repository gate passed, and an audit of prior CP1-CP10 artifacts found no prior `root_seed` use of the reserved range. The confirmation protocol froze the pi2/pi4 model identities, MARGIN_GATED threshold 3.4007568359375, six-route CP7 structure, company_survival hero policy, CP7 termination semantics, exact two-sided McNemar test, paired-difference CI method, behavioral diagnostics, and classification rules. No learning, normalization fitting, model selection, threshold tuning, mechanic change, or production AI change occurred.

On 384 paired untouched company_survival routes, pi2 won 314 (81.77%) and pi4-conservative won 326 (84.90%). Discordant outcomes favored pi4 20 to 8; the paired difference was +3.125 percentage points with 95% paired CI +0.44 to +5.81 points and exact two-sided McNemar p=0.0357. Under the frozen rule this is `CLEARLY_CONFIRMED`. The effect is smaller than CP10's +6.25-point estimate but directionally consistent and broadly compatible with its uncertainty. Pi4 remained behaviorally distinct from pi2 with a 27.53% override rate, kept Mark far below the pathological pi3 regime, replicated substantially elevated baseline Grab selection without G1, and retained Bite/direct-pressure behavior. Effort Drain was selected on all 347 legal pi4 opportunities, strengthening the `PERSISTENT_POTENTIAL_DOMINANCE` research flag. Clear confirmation makes pi4 scientifically eligible for a separate production-evaluation study, not automatic promotion. No production balance change is justified. The frozen next broad research direction is `EFFORT_DRAIN_CAUSAL_STUDY`.

A post-seed-access closeout bug affected only the optional pooled-summary helper: it indexed CP10's sealed paired-comparison JSON one level too shallow after raw CP11 simulation, primary statistics, classification, route robustness, support, and behavioral artifacts were already written. The untouched routes were not rerun. `analysis_correction.json` records the reporting-only fix, and the pooled/production/recommendation/report outputs were recomputed solely from the sealed summaries.
