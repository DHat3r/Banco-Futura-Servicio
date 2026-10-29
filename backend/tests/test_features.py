"""Tests de feature engineering (capa Silver en pandas). No necesitan la API."""

import pandas as pd
import pytest

from app.features import ALL_FEATURES, add_silver_features, build_inference_row


def _fila(**campos) -> pd.DataFrame:
    """Fila base sin ninguna bandera de riesgo activa; los campos pasados la sobrescriben."""
    base = {
        "Tenure": 12,
        "support_calls": 0,
        "total_spend": 300.0,
        "payment_delay": 0,
        "last_interaction": 0,
    }
    base.update(campos)
    return pd.DataFrame([base])


def test_support_calls_per_tenure_con_tenure_cero_usa_el_valor_crudo():
    df = add_silver_features(_fila(Tenure=0, support_calls=5))

    assert df.loc[0, "support_calls_per_tenure"] == 5


def test_support_calls_per_tenure_con_tenure_positivo_divide():
    df = add_silver_features(_fila(Tenure=3, support_calls=6))

    assert df.loc[0, "support_calls_per_tenure"] == pytest.approx(2.0)


def test_spend_per_tenure_con_tenure_cero_usa_el_valor_crudo():
    df = add_silver_features(_fila(Tenure=0, total_spend=250.0))

    assert df.loc[0, "spend_per_tenure"] == pytest.approx(250.0)


def test_spend_per_tenure_con_tenure_positivo_divide():
    df = add_silver_features(_fila(Tenure=4, total_spend=300.0))

    assert df.loc[0, "spend_per_tenure"] == pytest.approx(75.0)


def test_risk_score_flags_suma_las_cuatro_banderas():
    df = add_silver_features(
        _fila(Tenure=3, support_calls=4, payment_delay=15, last_interaction=21)
    )

    assert df.loc[0, "risk_score_flags"] == 4


def test_risk_score_flags_cero_si_ninguna_bandera_se_activa():
    df = add_silver_features(_fila())

    assert df.loc[0, "risk_score_flags"] == 0


@pytest.mark.parametrize(
    ("campo", "valor_que_suma", "valor_que_no_suma"),
    [
        ("Tenure", 6, 7),
        ("support_calls", 4, 3),
        ("payment_delay", 15, 14),
        ("last_interaction", 21, 20),
    ],
)
def test_risk_score_flags_respeta_los_limites(campo, valor_que_suma, valor_que_no_suma):
    con_limite = add_silver_features(_fila(**{campo: valor_que_suma}))
    bajo_limite = add_silver_features(_fila(**{campo: valor_que_no_suma}))

    assert con_limite.loc[0, "risk_score_flags"] == 1
    assert bajo_limite.loc[0, "risk_score_flags"] == 0


def test_build_inference_row_devuelve_exactamente_all_features():
    cliente = {
        "Age": 42,
        "Tenure": 5,
        "usage_frequency": 8,
        "support_calls": 6,
        "payment_delay": 22,
        "total_spend": 480.0,
        "last_interaction": 35,
        "Gender": "Female",
        "subscription_type": "Basic",
        "contract_length": "Monthly",
    }

    fila = build_inference_row(cliente)

    assert len(fila) == 1
    assert list(fila.columns) == ALL_FEATURES
