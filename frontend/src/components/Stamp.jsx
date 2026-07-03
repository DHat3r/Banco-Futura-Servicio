const TONOS = {
  bajo: { label: "Riesgo bajo" },
  medio: { label: "Riesgo medio" },
  alto: { label: "Riesgo alto" },
  positivo: { label: "Sentimiento positivo" },
  negativo: { label: "Sentimiento negativo" },
};

/**
 * Sello tipo "timbre de expediente" — el elemento de marca de la mesa de
 * riesgo. Se usa tanto para nivel de riesgo de churn como para sentimiento.
 */
export default function Stamp({ tone, value, size = "lg" }) {
  const meta = TONOS[tone] || { label: value };

  return (
    <div className={`stamp stamp--${tone} stamp--${size}`} role="img" aria-label={meta.label}>
      <span className="stamp__ring" aria-hidden="true" />
      <span className="stamp__value">{value}</span>
    </div>
  );
}
