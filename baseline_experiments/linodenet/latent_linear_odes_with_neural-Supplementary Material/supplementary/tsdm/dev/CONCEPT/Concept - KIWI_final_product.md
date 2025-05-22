# Kiwi Final Product Data Pipeline

Task Formulation:

- Following the general Time Series Task formulation, the task can be described as filling in
  a single missing value in the metadata.
- Model input: slice of TS, optionally metadata.
- Encoder needed for both input and target.
  - Encoder should work during inference, when run_id, exp_id is unknown.
  - Encoder must work with float timestamps on a seconds scale.
