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
