// Fallback — se usa mientras carga /metadata/cliente o si la API no responde.
// La fuente de verdad real vive en backend/app/features.py.
export const FALLBACK_GENDER_OPTIONS = ["Female", "Male"];
export const FALLBACK_SUBSCRIPTION_OPTIONS = ["Basic", "Standard", "Premium"];
export const FALLBACK_CONTRACT_OPTIONS = ["Monthly", "Quarterly", "Annual"];

export const ETIQUETAS_GENERO = { Female: "Femenino", Male: "Masculino" };
export const ETIQUETAS_CONTRATO = { Monthly: "Mensual", Quarterly: "Trimestral", Annual: "Anual" };

export const VALORES_INICIALES_CLIENTE = {
  Age: 35,
  Tenure: 12,
  usage_frequency: 10,
  support_calls: 2,
  payment_delay: 5,
  total_spend: 600,
  last_interaction: 10,
  Gender: "Female",
  subscription_type: "Standard",
  contract_length: "Annual",
};
