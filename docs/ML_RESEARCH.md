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
