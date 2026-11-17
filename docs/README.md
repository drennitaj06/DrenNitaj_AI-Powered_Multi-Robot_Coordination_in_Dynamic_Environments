# Project Documentation

Documentation for *AI-Powered Multi-Robot Coordination in Dynamic Environments* (Design and Analysis of Algorithms). Each topic has exactly one source of truth; other documents link to it instead of repeating it.

| Document | Source of truth for |
|---|---|
| [01-requirements.md](01-requirements.md) | Specification, functional/non-functional requirements, inputs/outputs, assumptions, acceptance criteria, traceability |
| [02-mathematical-model.md](02-mathematical-model.md) | Formal model: grid, dynamics, conflicts and arbitration, tasks and priorities, planning constraints, reference vs executed motion, Dec-POMDP, rewards, assignment problem, metrics, **summary of guarantees (§12)** |
| [03-architecture.md](03-architecture.md) | Components, interfaces, simulation pipeline, technology stack, design decisions (ADRs), repository layout, conventions |
| [04-algorithms.md](04-algorithms.md) | Algorithm design: pseudocode, data structures, correctness, complexity, edge cases, per-algorithm tests |
| [05-testing-and-evaluation.md](05-testing-and-evaluation.md) | Evidence levels, test strategy and gates, experiment protocol and budgets (E1–E7), statistics, hypotheses, results register |
| [06-kaggle-datasets.md](06-kaggle-datasets.md) | Dataset research, provisional selection, verification gate, licences, data policy, integration pipeline |
| [07-development-plan.md](07-development-plan.md) | GitHub workflow, the 8 issues, dependencies, resolved approval decisions, risks |
| [verification-log.md](verification-log.md) | Chronological record of executed tests and reviews |

Suggested reading order: 01 → 02 → 03 → 04, then 05–07 as needed.
