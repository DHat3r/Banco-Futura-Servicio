"""
Descarga reseñas de apps de bancos desde Google Play y las etiqueta por estrellas
(supervisión débil) para entrenar el modelo de sentimiento.

Uso (desde backend/):
    python scripts/review_scraper.py              # descarga y guarda data/resenas_bancos.csv
    python scripts/review_scraper.py --verificar  # solo comprueba que los ids existan
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from google_play_scraper import Sort, app, reviews

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SALIDA = DATA_DIR / "resenas_bancos.csv"

# (id de la app, país de la tienda): cada banco se consulta en la tienda donde están sus clientes.
APPS = [
    ("net.veritran.becl.prod", "cl"),                # BancoEstado
    ("cl.android", "cl"),                            # Banco Falabella Chile
    ("com.mercadopago.wallet", "cl"),                # Mercado Pago
    ("cl.bancochile.mi_banco", "cl"),                # Banco de Chile
    ("cl.bci.app.personas", "cl"),                   # Bci
    ("es.bancosantander.apps", "es"),                # Santander España
    ("com.konylabs.ItauMobileBank", "cl"),           # Itaú Chile
    ("dif.tech.plata", "mx"),                        # Banco Plata (México)
    ("com.mosync.app_Banco_Galicia", "ar"),          # Galicia (Argentina)
    ("es.lacaixa.mobile.android.newwapicon", "es"),  # CaixaBank
]
RESENAS_POR_APP = 2000
LARGO_MINIMO = 10


def etiquetar(estrellas: int) -> int | None:
    """1-2 estrellas -> 0 (negativo), 4-5 -> 1 (positivo), 3 -> None (se descarta)."""
    if estrellas <= 2:
        return 0
    if estrellas >= 4:
        return 1
    return None


def descargar_resenas(app_id: str, pais: str, cantidad: int) -> list[dict]:
    """Devuelve una lista de {"texto": ..., "estrellas": ..., "app": ...} SIN datos personales."""
    resultados, _ = reviews(app_id, lang="es", country=pais, sort=Sort.NEWEST, count=cantidad)
    # Solo texto y estrellas: userName, userImage y demás datos del autor no se guardan.
    return [
        {"texto": r["content"], "estrellas": r["score"], "app": app_id}
        for r in resultados
        if r["content"]
    ]


def verificar_apps() -> None:
    """Imprime título y desarrollador de cada app, para confirmar que los ids son correctos."""
    for app_id, pais in APPS:
        try:
            info = app(app_id, lang="es", country=pais)
            print(info["title"], "|", info["developer"])
        except Exception as e:  # noqa: BLE001 - solo es un chequeo manual
            print(app_id, "falló:", e)


def construir_dataset(filas: list[dict]) -> pd.DataFrame:
    """Etiqueta, descarta las 3 estrellas y limpia duplicados y textos muy cortos."""
    df = pd.DataFrame(filas, columns=["texto", "estrellas", "app"])
    df["texto"] = df["texto"].str.strip()
    df["etiqueta"] = df["estrellas"].map(etiquetar)
    df = df.dropna(subset=["etiqueta"])
    df["etiqueta"] = df["etiqueta"].astype(int)
    df = df.drop_duplicates(subset=["texto"])
    return df[df["texto"].str.len() >= LARGO_MINIMO]


def main() -> None:
    filas: list[dict] = []
    for app_id, pais in APPS:
        try:
            resenas = descargar_resenas(app_id, pais, RESENAS_POR_APP)
        except Exception as e:  # noqa: BLE001 - una app caída no debe frenar las demás
            print(f"{app_id}: falló ({e})")
            continue
        print(f"{app_id}: {len(resenas)} reseñas")
        filas.extend(resenas)

    df = construir_dataset(filas)

    print("\nPor etiqueta (0 = negativo, 1 = positivo):")
    print(df["etiqueta"].value_counts().to_string())
    print("\nPor app:")
    print(df["app"].value_counts().to_string())

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SALIDA, index=False)
    print(f"\nGuardado en {SALIDA} ({len(df)} reseñas)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verificar", action="store_true", help="Solo comprueba los ids de APPS")
    args = parser.parse_args()
    verificar_apps() if args.verificar else main()
