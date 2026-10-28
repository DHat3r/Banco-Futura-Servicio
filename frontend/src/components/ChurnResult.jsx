import Stamp from "./Stamp.jsx";

const TONO_POR_NIVEL = { BAJO: "bajo", MEDIO: "medio", ALTO: "alto" };

export default function ChurnResult({ resultado, error }) {
  return (
    <div className="panel panel--result">
      <div className="panel__eyebrow">Resultado</div>
      <h2 className="panel__title">Ficha de riesgo</h2>

      {error && <p className="error-text">{error}</p>}

      {!resultado && !error && (
        <p className="panel__hint">Completa el formulario y evalúa para ver el resultado acá.</p>
      )}

      {resultado && (
        <div className="result fade-in">
          <div className="result__top">
            <Stamp tone={TONO_POR_NIVEL[resultado.nivel_riesgo]} value={resultado.nivel_riesgo} />
            <div className="result__prob">
              <span className="result__prob-value">{Math.round(resultado.prob_churn * 100)}%</span>
              <span className="result__prob-label">probabilidad de abandono</span>
            </div>
          </div>

          <h3 className="result__subtitle">Factores que más influyeron</h3>
          <ul className="ledger">
            {resultado.factores_principales.map((f) => (
              <li className="ledger__row" key={f.feature}>
                <span className="ledger__label">{f.feature_label}</span>
                <span className="ledger__dots" aria-hidden="true" />
                <span className="ledger__value">
                  {f.valor_cliente}
                  <span className={`ledger__impact ${f.impacto >= 0 ? "ledger__impact--up" : "ledger__impact--down"}`}>
                    {f.impacto >= 0 ? "▲" : "▼"} {Math.abs(f.impacto).toFixed(2)}
                  </span>
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
