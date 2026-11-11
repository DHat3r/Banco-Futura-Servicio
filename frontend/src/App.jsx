import { useEffect, useState } from "react";
import ClientForm from "./components/ClientForm.jsx";
import ChurnResult from "./components/ChurnResult.jsx";
import SentimentPanel from "./components/SentimentPanel.jsx";
import { obtenerEstadoAPI, predecirChurn } from "./api.js";
import ClientePanel from "./components/ClientePanel.jsx";

// En el plan gratis de Render la API se duerme sin visitas y tarda cerca de un minuto en
// despertar. Por eso /health se consulta seguido mientras no responde y espaciado cuando ya
// responde (solo para notar si se cae). Los primeros 3 minutos de fallas se muestran como
// "despertando"; después, como "no disponible", pero se sigue intentando.
const CADA_MS_DESCONECTADA = 10_000;
const CADA_MS_CONECTADA = 3 * 60_000;
const FALLAS_DESPERTANDO = 18; // 18 x 10 s = 3 minutos

export default function App() {
  const [resultadoChurn, setResultadoChurn] = useState(null);
  const [errorChurn, setErrorChurn] = useState(null);
  const [loadingChurn, setLoadingChurn] = useState(false);
  const [estadoAPI, setEstadoAPI] = useState(null);
  const [despertando, setDespertando] = useState(false);

  useEffect(() => {
    let cancelado = false;
    let temporizador;

    async function consultar(fallasSeguidas) {
      try {
        const estado = await obtenerEstadoAPI();
        if (cancelado) return;
        setEstadoAPI(estado);
        setDespertando(false);
        temporizador = setTimeout(() => consultar(0), CADA_MS_CONECTADA);
      } catch {
        if (cancelado) return;
        const fallas = fallasSeguidas + 1;
        setEstadoAPI({ status: "down" });
        setDespertando(fallas <= FALLAS_DESPERTANDO);
        temporizador = setTimeout(() => consultar(fallas), CADA_MS_DESCONECTADA);
      }
    }

    consultar(0);
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
      <ClientePanel apiArriba={apiArriba} />
      <footer className="footer">
        Proyecto académico · Equipo 4 · Data Driven Banking
      </footer>
    </div>
  );
}
