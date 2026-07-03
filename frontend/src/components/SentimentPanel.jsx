import { useState } from "react";
import Stamp from "./Stamp.jsx";
import { predecirSentimiento } from "../api.js";

export default function SentimentPanel() {
  const [texto, setTexto] = useState("");
  const [resultado, setResultado] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function analizar(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const data = await predecirSentimiento(texto);
      setResultado(data);
    } catch (err) {
      setResultado(null);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <form className="panel panel--wide" onSubmit={analizar}>
      <div className="panel__eyebrow">Expediente 02</div>
      <h2 className="panel__title">Analizar mensaje de un cliente</h2>
      <p className="panel__hint">
        Pega un reclamo o mensaje de atención al cliente para ver su sentimiento. Útil para complementar la ficha
        de riesgo: un cliente de alto riesgo con un reclamo reciente negativo es prioridad de contacto.
      </p>

      <textarea
        className="textarea"
        rows={3}
        placeholder="Ej: Llevo dos semanas esperando que me respondan el reclamo…"
        value={texto}
        onChange={(e) => setTexto(e.target.value)}
        required
      />

      <div className="sentiment-row">
        <button type="submit" className="btn btn--secondary" disabled={loading || texto.trim().length < 3}>
          {loading ? "Analizando…" : "Analizar sentimiento"}
        </button>

        {error && <p className="error-text">{error}</p>}

        {resultado && (
          <div className="sentiment-result fade-in">
            <Stamp tone={resultado.sentimiento} value={resultado.sentimiento} size="sm" />
            <span className="sentiment-result__score">
              score {resultado.score.toFixed(2)}
            </span>
          </div>
        )}
      </div>
    </form>
  );
}
