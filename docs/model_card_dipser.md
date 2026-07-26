# DIPSER recognition model card

## Intended task

Predict publisher-expert attention bands (`low / medium / high`) from DIPSER RGB-derived pose metadata and wearable sensor summaries. This is not Teaching Skill quality, emotion diagnosis, identity recognition, grading, or an end-to-end raw-video model.

## Development results

- Strict causal session SGKF5: Accuracy `0.8176`, Macro-F1 `0.6357`.
- Strict causal leave-one-session-out: Accuracy `0.8176`, Macro-F1 `0.6249`.
- Offline full-session transductive SGKF5 candidate: Accuracy `0.9020`, Macro-F1 `0.7435`.
- Offline full-session LOSO: Accuracy `0.8986`, Macro-F1 `0.7397`.
- Fifty-seed mean Accuracy: `0.8883`.
- Fixed LOCO / LOAO / cohort×activity blocking Accuracy: `0.6554 / 0.8919 / 0.6351`.

The 0.902 candidate uses complete held-out-session context, including future windows and other participants. It is post-selection, retrospective, single-site, and unsuitable for real-time or single-person deployment claims. Its low-class recall and balanced accuracy are materially below its sample-weighted Accuracy.

## Required reporting

Always report Accuracy, Macro-F1, balanced accuracy, per-class precision/recall/support, session-cluster intervals, complete-case coverage, causal vs full-session counterfactuals, cross-seed sensitivity, and block stress tests.

## Deployment gate

Deployment accuracy can be established only by a v3 frozen checkpoint evaluated once on an identity/content-disjoint, target-site, prospective lockbox. The checkpoint binds the strict training feature bundle, ClaimContract, class schema, and claim-cluster field. The external dataset, feature bundle, and coverage denominator must be registered before evaluation, signed with an externally governed Ed25519 key, and atomically consumed once. Every checkpoint-bound metric, cluster-CI, per-class, support, coverage, prospective, and independence gate must pass. A failed gate remains a valid negative result and must not be tuned on the lockbox.

A valid signature proves integrity and possession of the corresponding key; it does not by itself prove organizational independence. The trusted public key must be pinned through governance outside the model-development team before results are inspected.

## Prohibited uses

Do not use this model as the sole basis for grading, discipline, surveillance, diagnosis, disability inference, employment, access control, or teacher/student ranking.
