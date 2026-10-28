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
  const [loading, setLoading] = useState(false);

  async function buscarCliente() {
    setLoading(true);
    setError(null);
    setChurnData(null);
    setMensajes([]);
    try {
      const data = await churnPorId(Number(idCliente));
      setChurnData(data);
      const msgs = await obtenerMensajesCliente(Number(idCliente));
      setMensajes(msgs.analisis || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function enviarMensaje() {
    if (!nuevoMensaje.trim()) return;
    try {
      await agregarMensaje(Number(idCliente), nuevoMensaje);
      setNuevoMensaje("");
      const msgs = await obtenerMensajesCliente(Number(idCliente));
      setMensajes(msgs.analisis || []);
    } catch (err) {
      setError(err.message);
    }
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

          {mensajes.length === 0 && (
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
