# RecBole → Hopwise Model Comparison Report

## Summary

| Category | Hopwise | RecBole | Missing | Extra in Hopwise |
|----------|---------|---------|---------|------------------|
| context_aware_recommender | 18 | 18 | 0 | 0 |
| exlib_recommender | 0 | 0 | 0 | 0 |
| general_recommender | 31 | 32 | 1 | 0 |
| knowledge_aware_recommender | 15 | 10 | 0 | 5 |
| knowledge_graph_embedding_recommender | 14 | 0 | 0 | 14 |
| path_language_modeling_recommender | 5 | 0 | 0 | 5 |
| sequential_recommender | 31 | 31 | 0 | 0 |

**Total missing models: 1**

## Missing Models by Category

### general_recommender

| Model | Complexity | Notes |
|-------|------------|-------|
| AsymKNN | 🟡 Medium | Deps: scipy; ~195 LOC |

## Integration Checklist

For each model:

1. Copy `recbole/model/<category>/<model>.py` → `hopwise/model/<category>/<model>.py`
2. Replace imports: `recbole.` → `hopwise.`
3. Copy `recbole/properties/model/<Model>.yaml` → `hopwise/properties/model/<Model>.yaml`
4. Add export to `hopwise/model/<category>/__init__.py`
5. Check if custom layers needed → add to `hopwise/model/layers.py`
6. Check if custom losses needed → add to `hopwise/model/loss.py`
7. Test: `python -m hopwise --model=<Model> --dataset=ml-100k`
