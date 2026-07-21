# Local Data Directory

Datasets and generated splits belong here, but they are intentionally excluded from Git. Each collaborator supplies and manages their own data locally.

The example configs expect:

- `data/splits/train.csv`
- `data/splits/validation.csv`
- `data/splits/test.csv`

Create those files with `python ipi.py split-data` or update the paths in the example configs. See [`docs/dataset_format.md`](../docs/dataset_format.md) for the input schema.

Do not force-add patient, confidential, or licensed data to Git.
