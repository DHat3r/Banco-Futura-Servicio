const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function postJSON(path, body) {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const data = await res.json().catch(() => null);

  if (!res.ok) {
    const detail = data?.detail || `Error ${res.status}`;
    throw new Error(detail);
  }
  return data;
}

export function predecirChurn(cliente) {
  return postJSON("/predict/churn", cliente);
}

// modelo: "completo" (BETO en PyTorch + Keras) o "liviano" (BETO int8 en ONNX).
export function predecirSentimiento(texto, modelo = "completo") {
  const ruta = modelo === "liviano" ? "/predict/sentimiento/liviano" : "/predict/sentimiento";
  return postJSON(ruta, { texto });
}

export async function obtenerMetadataCliente() {
  const res = await fetch(`${API_URL}/metadata/cliente`);
  if (!res.ok) throw new Error("No se pudo cargar la metadata del cliente");
  return res.json();
}

// Si la API acepta la conexión pero no contesta (pasa mientras Render la reinicia), sin
// tiempo límite el fetch esperaría indefinidamente y App dejaría de reintentar.
const TIEMPO_LIMITE_HEALTH_MS = 15_000;

export async function obtenerEstadoAPI() {
  const res = await fetch(`${API_URL}/health`, {
    signal: AbortSignal.timeout(TIEMPO_LIMITE_HEALTH_MS),
    cache: "no-store", // el estado cambia: nunca usar una respuesta guardada
  });
  if (!res.ok) throw new Error("API no disponible");
  return res.json();
}

// Una página de la cartera de clientes; filtroId filtra los IDs que empiezan así.
export async function obtenerCartera(pagina, filtroId = "") {
  const params = new URLSearchParams({ pagina: String(pagina) });
  if (filtroId) params.set("id", filtroId);
  const res = await fetch(`${API_URL}/clientes?${params}`);
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new Error(`No se pudo cargar la cartera (error ${res.status})`);
  return data;
}

export async function churnPorId(id_cliente) {
  const res = await fetch(`${API_URL}/cliente/${id_cliente}/churn`);
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new Error(data?.detail || `Error ${res.status}`);
  return data;
}

export async function agregarMensaje(id_cliente, texto) {
  const res = await fetch(`${API_URL}/cliente/${id_cliente}/mensaje`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ texto }),
  });
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new Error(data?.detail || `Error ${res.status}`);
  return data;
}

export async function obtenerMensajesCliente(id_cliente) {
  const res = await fetch(`${API_URL}/cliente/${id_cliente}/mensajes`);
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new Error(data?.detail || `Error ${res.status}`);
  return data;
}