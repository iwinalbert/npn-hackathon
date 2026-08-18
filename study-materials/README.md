# study-materials

Role guides for the team. Learning material — **not** operational documentation.

Each role folder holds two guides covering the same ground at different depths:

| File | For |
|---|---|
| `easy-guide.pdf` | Someone new to the role. Plain language, analogies, no assumed background. |
| `detailed-guide.pdf` | Someone doing the role. Concepts tied to the actual files in this repository. |

## Roles

| Folder | Role | Working assets |
|---|---|---|
| [`backend-developer/`](backend-developer/) | The FastAPI service | [`backend/`](../backend/) |
| [`data-engineer/`](data-engineer/) | Raw data to a modelling panel | [`research/`](../research/) |
| [`devops-engineer/`](devops-engineer/) | Build, ship, run and debug the system | [`infra/`](../infra/) |
| [`feature-engineer/`](feature-engineer/) | Feature design and leakage control | [`research/`](../research/) |
| [`forecasting-evaluation/`](forecasting-evaluation/) | Backtesting and accuracy | [`research/`](../research/) |
| [`frontend-developer/`](frontend-developer/) | The React app | [`frontend/`](../frontend/) |
| [`genai-developer/`](genai-developer/) | The AI forecast assistant | [`backend/`](../backend/) |
| [`ml-engineer/`](ml-engineer/) | Model training and the frozen champion | [`research/`](../research/) |

All eight roles have PDF guides, built from Markdown source with pandoc for
consistent formatting.

## The distinction worth keeping

**Study material explains *why*. Operational documentation tells you *what to
run*.** They rot at different rates and serve different moments — you read one
once, and the other at 2am when something is broken.

Keep them apart:

| | Lives in |
|---|---|
| Learning a role | `study-materials/<role>/` |
| Operating the system | `infra/docs/` |
| How the system was built | `docs/` |
