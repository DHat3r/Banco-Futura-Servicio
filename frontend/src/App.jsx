import { useEffect, useState } from "react";
import ClientForm from "./components/ClientForm.jsx";
import ChurnResult from "./components/ChurnResult.jsx";
import SentimentPanel from "./components/SentimentPanel.jsx";
import { obtenerEstadoAPI, predecirChurn } from "./api.js";
import ClientePanel from "./components/ClientePanel.jsx";

// En el plan gratis de Render la API se duerme sin visitas y tarda cerca de un minuto en
// despertar: si /health falla, se reintenta cada 10 s durante 3 minutos antes de rendirse.
const REINTENTO_MS = 10_000;
const MAX_INTENTOS = 18;

export default function App() {
  const [resultadoChurn, setResultadoChurn] = useState(null);
  const [errorChurn, setErrorChurn] = useState(null);
  const [loadingChurn, setLoadingChurn] = useState(false);
  const [estadoAPI, setEstadoAPI] = useState(null);
  const [despertando, setDespertando] = useState(false);

  useEffect(() => {
    let cancelado = false;
    let temporizador;

    async function consultar(intento) {
      try {
        const estado = await obtenerEstadoAPI();
        if (cancelado) return;
        setEstadoAPI(estado);
        setDespertando(false);
      } catch {
        if (cancelado) return;
        setEstadoAPI({ status: "down" });
        const quedanIntentos = intento < MAX_INTENTOS;
        setDespertando(quedanIntentos);
        if (quedanIntentos) temporizador = setTimeout(() => consultar(intento + 1), REINTENTO_MS);
      }
    }

    consultar(1);
    return () => {
      cancelado = true;
      clearTimeout(temporizador);
    };
  }, []);

  async function manejarEvaluacion(cliente) {
    setLoadingChurn(true);
    setErrorChurn(null);
    try {
      const data = await predecirChurn(cliente);
      setResultadoChurn(data);
    } catch (err) {
      setResultadoChurn(null);
      setErrorChurn(err.message);
    } finally {
      setLoadingChurn(false);
    }
  }

  const apiArriba = estadoAPI?.status === "ok";

  return (
    <div className="page">
      <header className="header">
        <div>
          <p className="header__eyebrow">Banco Futura</p>
          <h1 className="header__title">Mesa de riesgo de clientes</h1>
        </div>
        <div className="header__status">
          <span
            className={`status-dot ${apiArriba ? "status-dot--ok" : "status-dot--down"}`}
            aria-hidden="true"
          />
          {estadoAPI === null
            ? "Verificando API…"
            : apiArriba
              ? "API conectada"
              : despertando
                ? "Despertando la API…"
                : "API no disponible"}
        </div>
      </header>

      <main className="grid-main">
        <ClientForm onSubmit={manejarEvaluacion} loading={loadingChurn} />
        <ChurnResult resultado={resultadoChurn} error={errorChurn} />
      </main>

      <SentimentPanel estadoAPI={estadoAPI} />
      <ClientePanel />
      <footer className="footer">
        Proyecto académico · Equipo 4 · Data Driven Banking
      </footer>
    </div>
  );
}
