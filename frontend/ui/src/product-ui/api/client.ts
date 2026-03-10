/* ─── Causeway API Client ─── */

const BASE_URL = "";

async function request<T>(
  path: string,
  options?: RequestInit & { params?: Record<string, string> }
): Promise<T> {
  const { params, ...init } = options || {};

  let url = `${BASE_URL}${path}`;
  if (params) {
    const searchParams = new URLSearchParams(params);
    url += `?${searchParams.toString()}`;
  }

  const res = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  if (!res.ok) {
    const body = await res.text();
    throw new ApiError(res.status, body, path);
  }

  return res.json();
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public body: string,
    public path: string
  ) {
    super(`API ${status} on ${path}: ${body}`);
    this.name = "ApiError";
  }
}

/* ═══════════════════════════════════════════════════════════
   Types — Domain models (rich, used for graph rendering)
   ═══════════════════════════════════════════════════════════ */

export type IngestionStatus = "pending" | "indexing" | "indexed" | "failed";
export type EvidenceStrength = "strong" | "moderate" | "hypothesis" | "contested";
export type ModelStatus = "draft" | "review" | "active" | "deprecated";
export type EdgeStatus = "draft" | "grounded" | "rejected";
export type VariableType = "continuous" | "discrete" | "binary" | "categorical";
export type MeasurementStatus = "measured" | "observable" | "latent";
export type CausalRole = "treatment" | "outcome" | "confounder" | "mediator" | "instrumental" | "covariate" | "unknown";
export type ConfidenceLevel = "high" | "medium" | "low";
export type OperationalMode = "world_model_construction" | "decision_support";
export type ProtocolState = "idle" | "routing" | "wm_discovery_running" | "wm_review_pending" | "wm_active" | "decision_support_running" | "response_ready" | "error";

export interface HealthResponse {
  status: string;
  version?: string;
  timestamp?: string;
}

export interface MetricsResponse {
  uptime_seconds: number;
  request_count: number;
  error_count: number;
  [key: string]: unknown;
}

/* ─── Domain-level document model (used for local state in upload flow) ─── */
export interface DocumentMetadata {
  doc_id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  sha256: string;
  storage_uri: string;
  ingestion_status: IngestionStatus;
  pageindex_doc_id?: string;
  haystack_doc_ids?: string[];
}

/* ─── Domain models for causal graph rendering ─── */
export interface CausalVariable {
  variable_id: string;
  name: string;
  definition: string;
  type: VariableType;
  measurement_status: MeasurementStatus;
  data_source?: string;
  unit?: string;
  role: CausalRole;
}

export interface EdgeMetadata {
  mechanism: string;
  evidence_strength: EvidenceStrength;
  edge_status?: EdgeStatus;
  rejection_reason?: string;
  evidence_refs: string[];
  contradicting_refs: string[];
  assumptions: string[];
  conditions: string[];
  confidence: number;
  notes?: string;
}

export interface CausalEdge {
  from_var: string;
  to_var: string;
  metadata: EdgeMetadata;
}

export interface CausalPath {
  path: string[];
  edges: CausalEdge[];
  mechanism_chain: string;
  strength: string;
}

export interface WorldModelVersion {
  version_id: string;
  domain: string;
  description: string;
  variables: Record<string, CausalVariable>;
  edges: CausalEdge[];
  dag_json: Record<string, unknown>;
  created_at: string;
  created_by: string;
  approved_at?: string;
  approved_by?: string;
  status: ModelStatus;
  replaces_version?: string;
}

export interface DecisionRecommendation {
  recommendation: string;
  confidence: ConfidenceLevel;
  expected_outcome: string;
  causal_paths: CausalPath[];
  evidence_refs: string[];
  risks: string[];
  unmeasured_factors: string[];
  suggested_actions: string[];
  suggested_data_collection?: string[];
  reasoning_trace?: string;
}

export interface EscalationNotice {
  escalation_id: string;
  message: string;
  reason: string;
  original_query?: Record<string, unknown>;
  suggested_mode1_scope?: Record<string, unknown>;
  conflicts_detected?: Array<Record<string, unknown>>;
}

export interface Mode1StageResponse {
  stage: string;
  detail?: string;
}

export interface ProtocolStatus {
  state: ProtocolState;
  is_idle?: boolean;
  is_running?: boolean;
  is_waiting_review?: boolean;
  history?: Array<{ from_state: string; to_state: string; timestamp: string }>;
}

/* ═══════════════════════════════════════════════════════════
   API Response DTOs — match backend Pydantic response models
   ═══════════════════════════════════════════════════════════ */

/** POST /api/v1/uploads → DocumentResponse */
export interface DocumentResponseDTO {
  doc_id: string;
  filename: string;
  content_hash: string;
  storage_uri: string;
  status: string;
  created_at: string;
}

/** GET /api/v1/documents → DocumentListItem[] */
export interface DocumentListItem {
  doc_id: string;
  filename: string;
  status: string;
}

/** POST /api/v1/index/{doc_id} → IndexResponse */
export interface IndexResponseDTO {
  doc_id: string;
  status: string;
  message: string;
}

/** POST /api/v1/search → SearchResponse */
export interface SearchResultDTO {
  content: string;
  doc_id: string;
  doc_title: string;
  score: number;
  section?: string;
  page?: number;
}

export interface SearchResponseDTO {
  query: string;
  total_results: number;
  results: SearchResultDTO[];
}

/** POST /api/v1/mode1/run → Mode1Response */
export interface Mode1ResponseDTO {
  trace_id: string;
  domain: string;
  stage: string;
  variables_discovered: number;
  edges_created: number;
  evidence_linked: number;
  requires_review: boolean;
  error?: string;
}

/** POST /api/v1/mode2/run → Mode2Response */
export interface Mode2ResponseDTO {
  trace_id: string;
  query: string;
  stage: string;
  recommendation?: string;
  confidence?: string;
  model_used?: string;
  evidence_count: number;
  escalate_to_mode1: boolean;
  escalation_reason?: string;
  error?: string;
  // Rich audit / trace fields
  expected_outcome?: string;
  reasoning_trace?: string;
  causal_paths?: { path: string[]; mechanism_chain: string; strength: string }[];
  risks?: string[];
  suggested_actions?: string[];
  audit_log?: { action: string; data: Record<string, unknown>; timestamp: string }[];
}

/** GET /api/v1/world-models → WorldModelSummary[] */
export interface WorldModelSummary {
  domain: string;
  version_id?: string;
  node_count: number;
  edge_count: number;
  status: string;
  variables: string[];
}

/** GET /api/v1/world-models/{domain}/detail → WorldModelDetail */
export interface VariableDetail {
  variable_id: string;
  name: string;
  definition: string;
  var_type?: string;
  role?: string;
}

export interface EdgeDetail {
  from_var: string;
  to_var: string;
  mechanism: string;
  strength?: string;
  confidence?: number;
}

export interface WorldModelDetail {
  domain: string;
  version_id?: string;
  node_count: number;
  edge_count: number;
  status: string;
  variables: VariableDetail[];
  edges: EdgeDetail[];
}

/** PATCH /api/v1/world-models/{domain} — request & response */
export interface PatchWorldModelRequest {
  add_variables?: Array<{
    variable_id: string;
    name?: string;
    definition?: string;
    type?: string;
    measurement_status?: string;
    role?: string;
  }>;
  remove_variables?: string[];
  add_edges?: Array<{
    from_var: string;
    to_var: string;
    mechanism?: string;
    strength?: string;
    confidence?: number;
  }>;
  remove_edges?: Array<{ from_var: string; to_var: string }>;
  update_edges?: Array<{
    from_var: string;
    to_var: string;
    mechanism?: string;
    evidence_strength?: string;
    confidence?: number;
  }>;
}

export interface PatchWorldModelResponse {
  old_version_id: string;
  new_version_id: string;
  variables_added: number;
  variables_removed: number;
  edges_added: number;
  edges_removed: number;
  edges_updated: number;
  conflicts: string[];
}

/** POST /api/v1/world-models/bridge — request & response */
export interface BuildBridgeRequest {
  source_domain: string;
  target_domain: string;
  use_llm?: boolean;
}

export interface BridgeEdgeResponse {
  source_domain: string;
  source_var: string;
  target_domain: string;
  target_var: string;
  mechanism: string;
  strength: string;
  confidence: number;
}

export interface ConceptMappingResponse {
  source_var: string;
  target_var: string;
  similarity_score: number;
  mapping_rationale: string;
}

export interface BuildBridgeResponse {
  bridge_id: string;
  source_domain: string;
  target_domain: string;
  bridge_edges: BridgeEdgeResponse[];
  shared_concepts: ConceptMappingResponse[];
  status: string;
}

/** GET /api/v1/world-models/bridges → BridgeSummary[] */
export interface BridgeSummary {
  bridge_id: string;
  source_version_id: string;
  target_version_id: string;
  edge_count: number;
  concept_count: number;
  status: string;
  created_at?: string;
}

/** GET /api/v1/world-models/bridges/{bridge_id} → BridgeDetail */
export interface BridgeDetail {
  bridge_id: string;
  source_version_id: string;
  target_version_id: string;
  bridge_edges: BridgeEdgeResponse[];
  shared_concepts: ConceptMappingResponse[];
  status: string;
  created_at?: string;
  description?: string;
}

/** POST /api/v1/admin/purge-documents — request & response */
export interface PurgeRequest {
  confirm: boolean;
}

export interface PurgeResponse {
  success: boolean;
  documents_deleted: number;
  vectors_deleted: number;
  files_deleted: number;
  errors: string[];
  warnings: string[];
}

/** POST /api/v1/mode1/approve → ApprovalRequest */
export interface ApprovalRequest {
  domain: string;
  approved_by: string;
}

/** POST /api/v1/query → QueryResponse */
export interface QueryResponse {
  trace_id: string;
  routed_mode: string;
  confidence: number;
  route_reason: string;
  result: Record<string, unknown>;
  error?: string;
}

/* ═══════════════════════════════════════════════════════════
   Backward-compat aliases
   ═══════════════════════════════════════════════════════════ */
export type SearchResult = SearchResultDTO;

/* ═══════════════════════════════════════════════════════════
   API Endpoints — paths match backend prefix /api/v1
   ═══════════════════════════════════════════════════════════ */

export const api = {
  // ─── Health & Metrics (root-level, no /api/v1 prefix) ───
  health: () => request<HealthResponse>("/health"),
  metrics: () => request<MetricsResponse>("/metrics"),

  // ─── Documents ───
  uploadDocument: async (file: File, description?: string): Promise<DocumentResponseDTO> => {
    const formData = new FormData();
    formData.append("file", file);
    if (description) formData.append("description", description);
    const res = await fetch(`${BASE_URL}/api/v1/uploads`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new ApiError(res.status, await res.text(), "/api/v1/uploads");
    return res.json();
  },

  getDocument: (docId: string) =>
    request<DocumentResponseDTO>(`/api/v1/documents/${docId}`),

  listDocuments: () =>
    request<DocumentListItem[]>("/api/v1/documents"),

  indexDocument: (docId: string) =>
    request<IndexResponseDTO>(`/api/v1/index/${docId}`, {
      method: "POST",
      body: JSON.stringify({ doc_id: docId }),
    }),

  // ─── Search ───
  search: (query: string, maxResults: number = 5, docId?: string) =>
    request<SearchResponseDTO>("/api/v1/search", {
      method: "POST",
      body: JSON.stringify({ query, max_results: maxResults, doc_id: docId }),
    }),

  // ─── Mode 1 — World Model Construction ───
  executeMode1: (
    domain: string,
    initialQuery: string,
    maxVars?: number,
    maxEdges?: number,
    docIds?: string[]
  ) =>
    request<Mode1ResponseDTO>("/api/v1/mode1/run", {
      method: "POST",
      body: JSON.stringify({
        domain,
        initial_query: initialQuery,
        max_variables: maxVars,
        max_edges: maxEdges,
        doc_ids: docIds,
      }),
    }),

  getMode1Stage: () => request<Mode1StageResponse>("/api/v1/mode1/status"),

  approveModel: (domain: string, approvedBy: string) =>
    request<WorldModelSummary>("/api/v1/mode1/approve", {
      method: "POST",
      body: JSON.stringify({ domain, approved_by: approvedBy }),
    }),

  // ─── Mode 2 — Decision Support ───
  executeMode2: (query: string, domainHint?: string) =>
    request<Mode2ResponseDTO>("/api/v1/mode2/run", {
      method: "POST",
      body: JSON.stringify({ query, domain_hint: domainHint }),
    }),

  // ─── World Models ───
  listModels: () => request<WorldModelSummary[]>("/api/v1/world-models"),

  getModel: (domain: string) =>
    request<WorldModelSummary>(`/api/v1/world-models/${domain}`),

  getModelDetail: (domain: string) =>
    request<WorldModelDetail>(`/api/v1/world-models/${domain}/detail`),

  patchWorldModel: (domain: string, patch: PatchWorldModelRequest) =>
    request<PatchWorldModelResponse>(`/api/v1/world-models/${domain}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    }),

  // ─── Bridges ───
  listBridges: () => request<BridgeSummary[]>("/api/v1/world-models/bridges"),

  getBridge: (bridgeId: string) =>
    request<BridgeDetail>(`/api/v1/world-models/bridges/${bridgeId}`),

  buildBridge: (req: BuildBridgeRequest) =>
    request<BuildBridgeResponse>("/api/v1/world-models/bridge", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  // ─── Unified Query ───
  query: (query: string, sessionId?: string) =>
    request<QueryResponse>("/api/v1/query", {
      method: "POST",
      body: JSON.stringify({ query, session_id: sessionId }),
    }),

  // ─── Protocol ───
  protocolStatus: () => request<ProtocolStatus>("/api/v1/protocol/status"),

  // ─── Admin ───
  purgeDocuments: (confirm: boolean) =>
    request<PurgeResponse>("/api/v1/admin/purge-documents", {
      method: "POST",
      body: JSON.stringify({ confirm }),
    }),
};
