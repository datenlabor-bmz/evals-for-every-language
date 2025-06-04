# Contributing

## Adding a new benchmark dataset

- Create a new file in `evals/datasets/` for loading the dataset(s) for your task, ideally via HuggingFace's `datasets` library.
- Define the task and its evaluation metric(s) in `evals/tasks.py`.
- Update `evals/backend.py` and `frontend/src/components/ScoreColumns.js` to include the new task and its metrics.
- Submit a pull request.

## Adding a new model

You can submit requests [here](https://forms.gle/ckvY9pS7XLcHYnaV8), and we will take care of the rest.

## Adding a new language

We believe that we already have all of the languages. If this is not the case, open an issue!
