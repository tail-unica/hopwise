#!/bin/bash


python -m pytest -v -n auto tests/metrics
echo "metrics tests finished"

python -m pytest -v -n auto tests/config/test_config.py
python -m pytest -v -n 4 tests/config/test_overall.py
echo "config tests finished"

python -m pytest -v -n auto tests/evaluation_setting
echo "evaluation_setting tests finished"

python -m pytest -v -n 5 tests/model/test_model_auto.py
python -m pytest -v -n auto tests/model/test_model_manual.py
echo "model tests finished"

python -m pytest -v -n auto tests/data/test_dataset.py
python -m pytest -v -n auto tests/data/test_dataloader.py
python -m pytest -v -n auto tests/data/test_transform.py
echo "data tests finished"

python -m pytest -v tests/hyper_tuning/test_hyper_tuning.py
echo "hyper_tuning tests finished"