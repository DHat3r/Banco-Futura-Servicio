import { useState } from "react";
import { churnPorId, agregarMensaje, obtenerMensajesCliente } from "../api.js";
import ChurnResult from "./ChurnResult.jsx";
import Stamp from "./Stamp.jsx";

export default function ClientePanel() {
  const [idCliente, setIdCliente] = useState("");
  const [churnData, setChurnData] = useState(null);
  const [mensajes, setMensajes] = useState([]);
  const [nuevoMensaje, setNuevoMensaje] = useState("");
  const [error, setError] = useState(null);
  const [avisoMensajes, setAvisoMensajes] = useState(null);
  const [loading, setLoading] = useState(false);

  // El endpoint de mensajes puede responder 503 si el modelo de sentimiento
  // está desactivado; en ese caso solo mostramos un aviso y no perdemos el churn.
  async function cargarMensajes(id) {
    try {
      const msgs = await obtenerMensajesCliente(id);
      setMensajes(msgs.analisis || []);
      setAvisoMensajes(null);
    } catch (err) {
      setMensajes([]);
      setAvisoMensajes("Análisis de sentimiento no disponible: " + err.message);
    }
  }

  async function buscarCliente() {
    setLoading(true);
    setError(null);
    setAvisoMensajes(null);
    setChurnData(null);
    setMensajes([]);
    let data;
    try {
      data = await churnPorId(Number(idCliente));
    } catch (err) {
      setError(err.message);
      setLoading(false);
      return;
    }
    setChurnData(data);
    await cargarMensajes(Number(idCliente));
    setLoading(false);
  }

  async function enviarMensaje() {
    if (!nuevoMensaje.trim()) return;
    try {
      await agregarMensaje(Number(idCliente), nuevoMensaje);
      setNuevoMensaje("");
    } catch (err) {
      setError(err.message);
      return;
    }
    await cargarMensajes(Number(idCliente));
  }

  return (
    <div className="panel panel--wide">
      <div className="panel__eyebrow">Expediente 03</div>
      <h2 className="panel__title">Buscar cliente por ID</h2>

      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem" }}>
        <input
          type="number"
          placeholder="ID del cliente"
          value={idCliente}
          onChange={(e) => setIdCliente(e.target.value)}
          style={{ flex: 1 }}
        />
        <button
          className="btn btn--primary"
          onClick={buscarCliente}
          disabled={loading || !idCliente}
        >
          {loading ? "Buscando…" : "Buscar"}
        </button>
      </div>

      {error && <p className="error-text">{error}</p>}
      {churnData && <ChurnResult resultado={churnData} />}

      {churnData && (
        <div style={{ marginTop: "1.5rem" }}>
          <h3 className="result__subtitle">Mensajes del cliente</h3>
          {avisoMensajes && <p className="panel__hint">{avisoMensajes}</p>}

          <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem" }}>
            <input
              type="text"
              placeholder="Escribe un mensaje del cliente…"
              value={nuevoMensaje}
              onChange={(e) => setNuevoMensaje(e.target.value)}
              style={{ flex: 1 }}
            />
            <button
              className="btn btn--secondary"
              onClick={enviarMensaje}
              disabled={!nuevoMensaje.trim()}
            >
              Agregar
            </button>
          </div>

          {mensajes.length === 0 && !avisoMensajes && (
            <p className="panel__hint">Sin mensajes aún para este cliente.</p>
          )}

          <ul className="ledger">
            {mensajes.map((m, i) => (
              <li className="ledger__row" key={i}>
                <span className="ledger__label">{m.texto}</span>
                <Stamp tone={m.sentimiento} value={m.sentimiento} size="sm" />
                <span className="ledger__value">
                  score {m.score.toFixed(2)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
