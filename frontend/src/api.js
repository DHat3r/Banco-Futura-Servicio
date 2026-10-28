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

export function predecirSentimiento(texto) {
  return postJSON("/predict/sentimiento", { texto });
}

export async function obtenerMetadataCliente() {
  const res = await fetch(`${API_URL}/metadata/cliente`);
  if (!res.ok) throw new Error("No se pudo cargar la metadata del cliente");
  return res.json();
}

export async function obtenerEstadoAPI() {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error("API no disponible");
  return res.json();
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