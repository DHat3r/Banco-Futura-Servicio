import { useEffect, useState } from "react";
import {
  ETIQUETAS_CONTRATO,
  ETIQUETAS_GENERO,
  FALLBACK_CONTRACT_OPTIONS,
  FALLBACK_GENDER_OPTIONS,
  FALLBACK_SUBSCRIPTION_OPTIONS,
  VALORES_INICIALES_CLIENTE,
} from "../constants.js";
import { obtenerMetadataCliente } from "../api.js";

const CAMPOS_NUMERICOS = [
  { name: "Age", label: "Edad", min: 18, max: 100 },
  { name: "Tenure", label: "Antigüedad (meses)", min: 0, max: 120 },
  { name: "usage_frequency", label: "Frecuencia de uso mensual", min: 0, max: 60 },
  { name: "support_calls", label: "Llamadas a soporte", min: 0, max: 50 },
  { name: "payment_delay", label: "Atraso en pagos (días)", min: 0, max: 90 },
  { name: "total_spend", label: "Gasto total acumulado", min: 0, max: 100000, step: 10 },
  { name: "last_interaction", label: "Días desde última interacción", min: 0, max: 365 },
];

export default function ClientForm({ onSubmit, loading }) {
  const [valores, setValores] = useState(VALORES_INICIALES_CLIENTE);
  const [opciones, setOpciones] = useState({
    gender_options: FALLBACK_GENDER_OPTIONS,
    subscription_options: FALLBACK_SUBSCRIPTION_OPTIONS,
    contract_options: FALLBACK_CONTRACT_OPTIONS,
  });

  useEffect(() => {
    obtenerMetadataCliente()
      .then(setOpciones)
      .catch(() => {
        /* nos quedamos con los valores de fallback */
      });
  }, []);

  function actualizarCampo(name, value) {
    setValores((prev) => ({ ...prev, [name]: value }));
  }

  function manejarEnvio(e) {
    e.preventDefault();
    const payload = {
      ...valores,
      Age: Number(valores.Age),
      Tenure: Number(valores.Tenure),
      usage_frequency: Number(valores.usage_frequency),
      support_calls: Number(valores.support_calls),
      payment_delay: Number(valores.payment_delay),
      total_spend: Number(valores.total_spend),
      last_interaction: Number(valores.last_interaction),
    };
    onSubmit(payload);
  }

  return (
    <form className="panel" onSubmit={manejarEnvio}>
      <div className="panel__eyebrow">Expediente 01</div>
      <h2 className="panel__title">Evaluar cliente</h2>
      <p className="panel__hint">Completa los datos del cliente para estimar su riesgo de abandono.</p>

      <div className="field-grid">
        {CAMPOS_NUMERICOS.map((campo) => (
          <label className="field" key={campo.name}>
            <span className="field__label">{campo.label}</span>
            <input
              type="number"
              min={campo.min}
              max={campo.max}
              step={campo.step || 1}
              value={valores[campo.name]}
              onChange={(e) => actualizarCampo(campo.name, e.target.value)}
              required
            />
          </label>
        ))}

        <label className="field">
          <span className="field__label">Género</span>
          <select value={valores.Gender} onChange={(e) => actualizarCampo("Gender", e.target.value)}>
            {opciones.gender_options.map((g) => (
              <option key={g} value={g}>
                {ETIQUETAS_GENERO[g] || g}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span className="field__label">Tipo de suscripción</span>
          <select
            value={valores.subscription_type}
            onChange={(e) => actualizarCampo("subscription_type", e.target.value)}
          >
            {opciones.subscription_options.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span className="field__label">Duración del contrato</span>
          <select
            value={valores.contract_length}
            onChange={(e) => actualizarCampo("contract_length", e.target.value)}
          >
            {opciones.contract_options.map((c) => (
              <option key={c} value={c}>
                {ETIQUETAS_CONTRATO[c] || c}
              </option>
            ))}
          </select>
        </label>
      </div>

      <button type="submit" className="btn btn--primary" disabled={loading}>
        {loading ? "Evaluando…" : "Evaluar riesgo de churn"}
      </button>
    </form>
  );
}
