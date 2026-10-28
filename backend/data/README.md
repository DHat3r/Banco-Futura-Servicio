# Datos

Los CSV no se versionan en git (pesan ~25 MB). Descárgalos de Kaggle:
[`muhammadshahidazeem/customer-churn-dataset`](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset)

Deja aquí estos dos archivos:

- `customer_churn_dataset-training-master.csv`
- `customer_churn_dataset-testing-master.csv`

Solo hacen falta para reentrenar el modelo (`scripts/train_xgboost.py`). La API funciona sin ellos porque el modelo ya entrenado está en `artifacts/`.
