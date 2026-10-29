# Datos

Los CSV no se versionan en git (pesan ~25 MB). Descárgalos de Kaggle:
[`muhammadshahidazeem/customer-churn-dataset`](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset)

Deja aquí estos dos archivos:

- `customer_churn_dataset-training-master.csv`
- `customer_churn_dataset-testing-master.csv`

Solo hacen falta para reentrenar el modelo (`scripts/train_xgboost.py`). La API funciona sin ellos porque el modelo ya entrenado está en `artifacts/`.

## Reseñas para el modelo de sentimiento (`resenas_bancos.csv`)

No está en git. Se genera con `python scripts/review_scraper.py` (reseñas de Google Play de 10 apps de bancos, etiquetadas por estrellas) y la usa `scripts/train_sentiment.py`.

## Muestra de clientes (`sample_clientes.csv`)

`sample_clientes.csv` (300 clientes) sí está en git. La API la usa si no encuentra el CSV completo, así que la búsqueda por ID funciona al clonar el repo. Para regenerarla:

```bash
python scripts/make_sample_clientes.py
```
