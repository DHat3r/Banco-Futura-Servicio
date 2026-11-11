import { useEffect, useState } from "react";
import Stamp from "./Stamp.jsx";
import { predecirSentimiento } from "../api.js";

// Los dos modelos de sentimiento: el mismo BETO, completo o cuantizado a int8.
const MODELOS = [
  { id: "completo", nombre: "BETO completo", detalle: "PyTorch + Keras", clave: "sentiment_model_listo" },
  { id: "liviano", nombre: "BETO liviano", detalle: "int8 en ONNX", clave: "sentiment_liviano_listo" },
];

export default function SentimentPanel({ estadoAPI }) {
  const disponibles = MODELOS.filter((m) => estadoAPI?.[m.clave]);
  const [modelo, setModelo] = useState("completo");
  const [texto, setTexto] = useState("");
  const [resultado, setResultado] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  // Si el modelo elegido no está cargado (por ejemplo en Render), se pasa al que sí lo esté.
  useEffect(() => {
    if (disponibles.length && !disponibles.some((m) => m.id === modelo)) {
      setModelo(disponibles[0].id);
    }
  }, [disponibles, modelo]);

  async function analizar(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const data = await predecirSentimiento(texto, modelo);
      setResultado({ ...data, modelo });
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

      <fieldset className="selector-modelo">
        <legend className="field__label">Modelo</legend>
        {MODELOS.map((m) => {
          const activo = Boolean(estadoAPI?.[m.clave]);
          return (
            <label key={m.id} className={`selector-modelo__opcion${activo ? "" : " is-apagado"}`}>
              <input
                type="radio"
                name="modelo-sentimiento"
                value={m.id}
                checked={modelo === m.id}
                disabled={!activo}
                onChange={() => {
                  setModelo(m.id);
                  setResultado(null);
                }}
              />
              <span className="selector-modelo__nombre">{m.nombre}</span>
              <span className="selector-modelo__detalle">{activo ? m.detalle : "no cargado"}</span>
            </label>
          );
        })}
      </fieldset>

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
              score {resultado.score.toFixed(2)} · {resultado.modelo === "liviano" ? "BETO liviano" : "BETO completo"}
            </span>
          </div>
        )}
      </div>
    </form>
  );
}
