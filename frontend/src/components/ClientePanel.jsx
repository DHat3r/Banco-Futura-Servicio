import { useEffect, useState } from "react";
import { churnPorId, agregarMensaje, obtenerCartera, obtenerMensajesCliente } from "../api.js";
import { ETIQUETAS_CONTRATO } from "../constants.js";
import ChurnResult from "./ChurnResult.jsx";
import Stamp from "./Stamp.jsx";

const dinero = new Intl.NumberFormat("es-CL", { style: "currency", currency: "USD", maximumFractionDigits: 0 });

export default function ClientePanel({ apiArriba }) {
  const [idCliente, setIdCliente] = useState(null);
  const [filtro, setFiltro] = useState("");
  const [pagina, setPagina] = useState(1);
  const [cartera, setCartera] = useState(null);
  const [errorCartera, setErrorCartera] = useState(null);
  const [churnData, setChurnData] = useState(null);
  const [mensajes, setMensajes] = useState([]);
  const [nuevoMensaje, setNuevoMensaje] = useState("");
  const [error, setError] = useState(null);
  const [avisoMensajes, setAvisoMensajes] = useState(null);
  const [loading, setLoading] = useState(false);

  // La cartera se pide cuando la API ya respondió (si está despertando, fallaría).
  // El filtro espera 300 ms sin teclear antes de consultar, para no pedir una página por cada tecla.
  useEffect(() => {
    if (!apiArriba) return;
    let cancelado = false;
    const espera = setTimeout(
      () => {
        obtenerCartera(pagina, filtro)
          .then((data) => {
            if (cancelado) return;
            setCartera(data);
            setErrorCartera(null);
          })
          .catch((err) => {
            if (!cancelado) setErrorCartera(err.message);
          });
      },
      filtro ? 300 : 0,
    );
    return () => {
      cancelado = true;
      clearTimeout(espera);
    };
  }, [apiArriba, pagina, filtro]);

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

  async function seleccionarCliente(id) {
    setIdCliente(id);
    setLoading(true);
    setError(null);
    setAvisoMensajes(null);
    setChurnData(null);
    setMensajes([]);
    let data;
    try {
      data = await churnPorId(id);
    } catch (err) {
      setError(err.message);
      setLoading(false);
      return;
    }
    setChurnData(data);
    await cargarMensajes(id);
    setLoading(false);
  }

  async function enviarMensaje() {
    if (!nuevoMensaje.trim()) return;
    try {
      await agregarMensaje(idCliente, nuevoMensaje);
      setNuevoMensaje("");
    } catch (err) {
      setError(err.message);
      return;
    }
    await cargarMensajes(idCliente);
  }

  return (
    <div className="panel panel--wide">
      <div className="panel__eyebrow">Expediente 03</div>
      <h2 className="panel__title">Cartera de clientes</h2>
      <p className="panel__hint">Elige un cliente para ver su ficha de riesgo y sus mensajes.</p>

      <div className="busqueda">
        <input
          type="text"
          inputMode="numeric"
          placeholder="Filtrar por ID"
          aria-label="Filtrar clientes por ID"
          value={filtro}
          onChange={(e) => {
            setFiltro(e.target.value.replace(/\D/g, ""));
            setPagina(1);
          }}
        />
        {cartera && (
          <span className="cartera__total">
            {cartera.total.toLocaleString("es-CL")} {cartera.total === 1 ? "cliente" : "clientes"}
          </span>
        )}
      </div>

      {!apiArriba && <p className="panel__hint">La cartera aparece cuando la API esté conectada.</p>}
      {errorCartera && <p className="error-text">{errorCartera}</p>}

      {cartera && (
        <>
          <div className="cartera">
            <table className="cartera__tabla">
              <thead>
                <tr>
                  <th scope="col">ID</th>
                  <th scope="col" className="num">Edad</th>
                  <th scope="col">Suscripción</th>
                  <th scope="col">Contrato</th>
                  <th scope="col" className="num">Atraso</th>
                  <th scope="col" className="num">Gasto</th>
                </tr>
              </thead>
              <tbody>
                {cartera.clientes.map((c) => (
                  <tr
                    key={c.id}
                    className={c.id === idCliente ? "is-elegido" : undefined}
                    onClick={() => seleccionarCliente(c.id)}
                  >
                    <td>
                      {/* El botón hace la fila usable con teclado y lector de pantalla. */}
                      <button
                        type="button"
                        className="cartera__id"
                        aria-current={c.id === idCliente ? "true" : undefined}
                        onClick={(e) => {
                          e.stopPropagation();
                          seleccionarCliente(c.id);
                        }}
                      >
                        {c.id}
                      </button>
                    </td>
                    <td className="num">{c.edad}</td>
                    <td>{c.suscripcion}</td>
                    <td>{ETIQUETAS_CONTRATO[c.contrato] ?? c.contrato}</td>
                    <td className="num">{c.atraso_pago} d</td>
                    <td className="num">{dinero.format(c.gasto_total)}</td>
                  </tr>
                ))}
                {cartera.clientes.length === 0 && (
                  <tr>
                    <td colSpan={6} className="cartera__vacia">
                      Ningún cliente tiene un ID que empiece con {filtro}.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="paginacion">
            <button
              type="button"
              className="btn btn--secondary"
              disabled={pagina <= 1}
              onClick={() => setPagina(pagina - 1)}
            >
              Anterior
            </button>
            <span>
              Página {cartera.pagina.toLocaleString("es-CL")} de {cartera.total_paginas.toLocaleString("es-CL")}
            </span>
            <button
              type="button"
              className="btn btn--secondary"
              disabled={pagina >= cartera.total_paginas}
              onClick={() => setPagina(pagina + 1)}
            >
              Siguiente
            </button>
          </div>
        </>
      )}

      {loading && <p className="panel__hint">Cargando la ficha…</p>}
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
