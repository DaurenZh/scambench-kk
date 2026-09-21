# Three-class labelling report

Total records: 37421

## Label distribution

| Label | Records | Share |
|---|---:|---:|
| LEGITIMATE | 18664 | 49.9% |
| SUSPICIOUS | 10979 | 29.3% |
| SCAM | 7778 | 20.8% |

## Per split

| Split | LEGITIMATE | SCAM | SUSPICIOUS |
|---|---|---|---|
| test | 1869 | 749 | 1116 |
| train | 14928 | 6258 | 8765 |
| validation | 1867 | 771 | 1098 |

## Rule that produced each label

| Rule | Records |
|---|---:|
| `benign:normal_engagement` | 15121 |
| `verify_required:risk_present` | 10944 |
| `confirmed:defensive_action_required` | 7778 |
| `verify_required:no_risk_signal` | 3456 |
| `defensive_action_on_benign_scenario` | 87 |
| `conflict:benign_action_with_attack_flag` | 34 |
| `defensive_action_without_attack_flag` | 1 |

## Relationship to the original binary flag

The three-class label refines `should_trigger_scam_defense` rather than
contradicting it: no record flagged benign became SCAM, and no record flagged
as an attack became LEGITIMATE.

| `should_trigger_scam_defense` | LEGITIMATE | SCAM | SUSPICIOUS |
|---|---|---|---|
| False | 18664 | 0 | 2 |
| True | 0 | 7778 | 10977 |

## Risk indicators behind the SUSPICIOUS class

| Indicator family | SUSPICIOUS records carrying it |
|---|---:|
| social_engineering | 8762 |
| privilege_escalation | 8172 |
| instruction_override | 1351 |
| suspicious_link | 1154 |
| credential_request | 521 |
| impersonation | 411 |
| urgency_pressure | 340 |
| identity_inconsistency | 311 |
| financial_request | 214 |

Records with no indicator family resolved: 82 (0.7% of SUSPICIOUS).
Records flagged `needs_review`: 122.

## Language coverage

| Language | Records |
|---|---:|
| en | 20143 |
| es | 3095 |
| pt | 2581 |
| de | 1780 |
| fr | 1512 |
| ar | 987 |
| vi | 980 |
| ko | 945 |
| ru | 928 |
| th | 917 |
| zh | 903 |
| hi | 897 |
| tr | 877 |
| ja | 876 |
