import type {HarnessUsage} from "@/lib/harness-stream";

export type SessionStatus = "active" | "succeeded" | "terminated_unable";

export type LessonPhase =
  | "orientation"
  | "explanation"
  | "worked_example"
  | "guided_practice"
  | "verification"
  | "transfer";

export interface LessonProgress {
  phase: LessonPhase;
  phase_label: string;
  phase_index: number;
  phase_count: 6;
  phase_reason?: string;
  current_concept?: string;
}

export interface SessionListItem {
  id: string;
  title: string;
  learner: string;
  status: SessionStatus;
  round: number;
  group: "pinned" | "scheduled" | "recent";
}

export interface TeachingMessage {
  id: string;
  role: "learner" | "teacher" | "tool";
  body: string;
  createdAt: string;
  skill?: string;
  lessonPhase?: LessonProgress;
  toolLabel?: string;
  toolDetail?: string;
  streaming?: boolean;
  thinking?: string;
  webSearchUsed?: boolean;
  sources?: Array<{title: string; url: string}>;
  status?: TurnStatus;
  error?: string;
  retryable?: boolean;
  attachmentLabels?: string[];
}

export type TurnStatus = "queued" | "running" | "stopped" | "failed" | "completed";

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface LearningProjectSummary {
  project_id: string;
  title: string;
  description: string;
  status: "active" | "archived";
  pinned: boolean;
  updated_at: string;
  syllabus_count: number;
  teaching_session_count: number;
  resource_count: number;
  chat_thread_count: number;
  note_count: number;
}

export interface LearningProjectTrashItem {
  project_id: string;
  title: string;
  updated_at: string;
  recovery_token: string;
}

export interface LearningProjectDeletionReceipt {
  schema: "teaching_skill_miner.project_deletion_receipt.v1";
  receipt_id: string;
  project_id: string;
  deleted_at: string;
  request_token_sha256: string;
  target_set_sha256: string;
  deleted_counts: Record<string, number>;
  retained_shared_counts: Record<string, number>;
  storage_cleanup_state: "completed";
  content_retained_in_receipt: false;
  user_managed_export_copies_deleted: false;
  user_managed_export_copies_status: "not_verifiable_by_running_app";
  backup_scope: "configured_live_stores_only";
}

export interface LearningProjectChatMessage {
  message_id: string;
  role: "user" | "assistant" | "tool";
  content: string;
  status: TurnStatus;
  created_at: string;
  web_search_used: boolean;
  sources: Array<{title: string; url: string}>;
}

export interface LearningProjectChatThread {
  thread_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: LearningProjectChatMessage[];
}

export interface LearningProjectChatThreadSummary {
  thread_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  preview: string;
}

export interface LearningProjectNote {
  note_id: string;
  title: string;
  body: string;
  created_at: string;
  updated_at: string;
}

export interface LearningProjectReferenceItem {
  reference_id: string;
  available?: boolean;
  title?: string;
  description?: string;
  module_count?: number;
  status?: SessionStatus;
  rounds_completed?: number;
  lesson_progress?: LessonProgress;
  metadata?: TeachingResourceSummary;
  project_reference_count?: number;
  ownership?: "shared_content_addressed_library" | "project_exclusive_reference" | "configured_resource_store_unavailable";
  remove_semantics?: string;
}

export interface LearningProjectPage<T> {
  schema: "teaching_skill_miner.learning_project_page.v1";
  project_id: string;
  project_updated_at: string;
  section: "chat_threads" | "chat_messages" | "notes" | "syllabi" | "teaching_sessions" | "resources";
  query: string;
  items: T[];
  total: number;
  next_cursor: string | null;
}

export interface LearningProject {
  schema: "teaching_skill_miner.learning_project.v1";
  project_id: string;
  title: string;
  description: string;
  status: "active" | "archived";
  pinned: boolean;
  created_at: string;
  updated_at: string;
  syllabus_ids: string[];
  teaching_session_ids: string[];
  resource_ids: string[];
  chat_threads: LearningProjectChatThread[];
  notes: LearningProjectNote[];
  claim_boundary: Record<string, boolean>;
}

export interface QueuedChatPrompt {
  id: string;
  threadId: string;
  text: string;
  webSearch: boolean;
  status: "queued";
}

export interface ChatResponse {
  schema_version?: string;
  mode: "chat";
  message: string;
  provider?: string;
  model?: string | null;
  latency_ms?: number | null;
  usage?: HarnessUsage;
  web_search_requested?: boolean;
  web_search_used?: boolean;
  sources?: Array<{title: string; url: string}>;
}

export interface TeachingResourceSummary {
  schema?: string;
  resource_id: string;
  staged_resource_id?: string;
  display_name: string;
  resource_type?: "text" | "document" | "presentation" | "pdf" | "image_ocr" | string;
  mime_type?: string;
  byte_size?: number;
  content_sha256?: string;
  extracted_char_count?: number;
  original_extracted_char_count?: number;
  truncated?: boolean;
  page_count?: number | null;
  extraction_engine?: string;
  needs_review?: boolean;
  requires_confirmation?: boolean;
  raw_media_retained?: boolean;
  remote_media_sent?: boolean;
  original_resource_sha256?: string;
  evidence_contract?: TeachingResourceEvidenceContract;
  resource_review?: TeachingResourceReviewSummary;
  review_requirements?: TeachingResourceReviewRequirements;
  grading_evidence_allowed?: false;
  mastery_evidence_allowed?: false;
  [key: string]: unknown;
}

export interface TeachingResourceEvidenceLayer {
  layer_id: string;
  kind: string;
  evidence_locator?: string;
  status: string;
  semantic_understanding_established?: false;
  visual_verification_status?: string;
  grading_evidence_allowed?: false;
  mastery_evidence_allowed?: false;
  [key: string]: unknown;
}

export interface TeachingResourceEvidenceConflict {
  conflict_id: string;
  kind: string;
  description: string;
  evidence_locators?: string[];
  resolution_status?: string;
  [key: string]: unknown;
}

export interface TeachingResourceEvidenceContract {
  schema?: string;
  layers: TeachingResourceEvidenceLayer[];
  conflicts: TeachingResourceEvidenceConflict[];
  decision: string;
  transcription_is_semantic_understanding?: false;
  semantic_analysis_is_answer_correctness?: false;
  grading_evidence_allowed?: false;
  mastery_evidence_allowed?: false;
  [key: string]: unknown;
}

export interface TeachingResourceReviewSummary {
  reviewed: boolean;
  review_version: number;
  review_id: string | null;
  review_scope: "untrusted_teaching_context_only";
  semantic_understanding_established: false;
  grading_evidence_allowed: false;
  mastery_evidence_allowed: false;
}

export interface TeachingResourceReviewRequirements {
  schema: "teaching_skill_miner.resource_review_requirements.v1";
  original_resource_sha256: string;
  conflicts: TeachingResourceEvidenceConflict[];
  layers: TeachingResourceEvidenceLayer[];
  required_resolved_conflict_ids: string[];
  reviewable_layer_ids: string[];
  raw_media_included: false;
  answer_key_or_grading_authority_included: false;
}

export interface TeachingResourceReviewAttestations {
  compared_with_original_source: true;
  uncertainties_removed_or_explicit: true;
  not_an_answer_key: true;
  context_only: true;
}

export interface TeachingResourceReviewRequest {
  resource_id: string;
  staged_resource_id: string;
  content_sha256: string;
  expected_review_version: number;
  resource_review_idempotency_key: string;
  original_resource_sha256: string;
  reviewed_text: string;
  resolved_conflict_ids: string[];
  excluded_layer_ids: string[];
  attestations: TeachingResourceReviewAttestations;
  review_note: string;
}

export interface TeachingResourceReviewResponse {
  schema: "teaching_skill_miner.dashboard_resource_review.v1";
  resource: TeachingResourceSummary;
  session_use: {
    status: "eligible_untrusted_context";
    decision: string | null;
    reason: null;
    grading_evidence_allowed: false;
    mastery_evidence_allowed: false;
  };
  review_projection?: Record<string, unknown>;
  original_resource_immutable: true;
  raw_media_sent: false;
  review_scope: "untrusted_teaching_context_only";
  semantic_understanding_established: false;
  grading_evidence_allowed: false;
  mastery_evidence_allowed: false;
}

export interface ResourceUploadItem {
  localId: string;
  name: string;
  mimeType?: string;
  status: "extracting" | "ready" | "truncated" | "blocked" | "failed";
  detail?: string;
  resource?: TeachingResourceSummary;
  removable?: boolean;
}

export interface SyllabusLesson {
  lesson_id: string;
  title: string;
  objective?: string;
  summary?: string;
  duration_minutes?: number;
  knowledge_components?: string[];
  prerequisites?: string[];
  materials?: Record<string, string>;
  teaching_goal?: Record<string, unknown>;
  start_payload?: {
    goal?: Record<string, unknown>;
    staged_resource_ids?: string[];
    [key: string]: unknown;
  };
  order?: number;
  [key: string]: unknown;
}

export interface SyllabusModule {
  module_id: string;
  title: string;
  description?: string;
  lessons: SyllabusLesson[];
  order?: number;
  [key: string]: unknown;
}

export interface TeachingSyllabus {
  syllabus_id: string;
  title: string;
  description?: string;
  audience?: string;
  estimated_duration_minutes?: number;
  learning_objectives?: string[];
  prerequisites?: string[];
  status?: "draft" | "ready" | "archived" | string;
  source?: string | Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
  modules: SyllabusModule[];
  [key: string]: unknown;
}

export interface SyllabusEditableLesson {
  title: string;
  objective: string;
  summary: string;
  duration_minutes: number;
  knowledge_components: string[];
  materials: {example: string; practice: string; transfer_task: string};
}

export interface SyllabusEditableModule {
  title: string;
  description: string;
  lessons: SyllabusEditableLesson[];
}

export interface SyllabusEditableDraft {
  title: string;
  description: string;
  audience: string;
  estimated_duration_minutes: number;
  learning_objectives: string[];
  prerequisites: string[];
  modules: SyllabusEditableModule[];
}

export interface SyllabusRevisionProjection {
  revision_id: string;
  revision_number: number;
  syllabus_id: string;
  content_sha256: string;
  parent_revision_id: string | null;
  created_at_utc: string;
  change_summary: string;
  status: "published" | "draft" | "historical";
}

export interface SyllabusVersionFamily {
  schema: "teaching_skill_miner.syllabus_version_projection.v1";
  family_id: string;
  version: number;
  published_revision_id: string;
  revisions: SyllabusRevisionProjection[];
  actor_boundary: {
    actor_kind: "local_operator_not_authenticated";
    authenticated: false;
    teacher_identity_claimed: false;
  };
}

export interface CurriculumAuthorityProjection {
  family_id: string;
  version: number;
  status: "reviewed" | "sealed" | "revoked";
  review: {
    review_id: string;
    teacher_spec: Record<string, unknown>;
    teacher_spec_sha256: string;
  };
  seal: null | {curriculum_id: string};
  revocation: null | {reason_code: string};
  [key: string]: unknown;
}

export interface CurriculumBlueprintResponse {
  curriculum_blueprint: Record<string, unknown>;
  curriculum_authority: CurriculumAuthorityProjection | null;
  review_status: "authenticated_teacher_review_required" | "reviewed" | "sealed" | "revoked" | "stale" | string;
  authoritative_for_runtime_grading: boolean;
}

export type RemoteConsentPurpose =
  | "remote_chat"
  | "remote_teaching"
  | "remote_syllabus_generation"
  | "public_web_search"
  | "remote_visual_analysis";

export interface RemoteConsentPolicy {
  purpose: RemoteConsentPurpose;
  provider_id: string;
  processing_region: string;
  data_categories: string[];
  provider_retention_days: number;
  provider_policy: {
    policy_id: string;
    policy_version: string;
    policy_source: "deployment_operator_asserted_external_terms_not_repository_verified" | "local_unverified_test_or_standalone";
    deletion_status: "outside_service_control_subject_to_provider_policy" | "provider_documents_zero_retention";
    documentation_url: string | null;
  };
  provider_policy_sha256: string;
  subject_policy: {
    policy_id: string;
    policy_version: string;
    policy_source: "organization_oidc_or_roster_policy" | "local_unverified_self_declaration";
    likely_minor: boolean;
    guardian_or_school_policy: "not_required" | "verified_guardian" | "verified_school_policy";
    remote_processing_eligible: boolean;
  };
  subject_policy_sha256: string;
  remote_processing_eligible: boolean;
}

export interface RemoteConsentReceipt extends RemoteConsentPolicy {
  schema: "teaching_skill_miner.remote_consent_receipt.v1";
  consent_id: string;
  subject_id: string;
  policy_version: number;
  guardian_or_school_policy: "not_required" | "verified_guardian" | "verified_school_policy";
  status: "active" | "revoked";
  granted_at_utc: string;
  expires_at_utc: string;
  revoked_at_utc: string | null;
  revocation_reason_code: string | null;
  receipt_sha256: string;
  signature: string;
}

export interface RemoteConsentListResponse {
  schema: "teaching_skill_miner.dashboard_remote_consent_list.v1";
  policy_version: number;
  receipts: RemoteConsentReceipt[];
  legacy_browser_receipts_authoritative: false;
}

export interface SyllabusGeneratePayload {
  topic: string;
  audience?: string;
  objectives?: string[];
  duration_minutes?: number;
  source_resource_ids?: string[];
  remote_consent_id: string;
}

export interface SyllabusLessonStartPayload {
  goal: Record<string, unknown>;
  syllabus_ref?: {
    syllabus_id?: string;
    module_id?: string;
    lesson_id?: string;
    content_sha256?: string;
    [key: string]: unknown;
  };
  staged_resource_ids?: string[];
  [key: string]: unknown;
}

export interface BootstrapPayload {
  schema_version?: string;
  mode?: string;
  cache_scope?: string;
  account_data_rights?: {
    mode: "authenticated_account_authority" | "local_only_no_account_authority";
    export?: string;
    deletion_prepare?: string;
    deletion_confirm?: string;
    deletion_resume?: string;
    deletion_status?: string;
    step_up?: string;
    recent_auth_required: boolean;
    recent_auth_satisfied?: boolean;
    minimum_assurance_level?: number;
    remote_provider_copies_deleted: false;
  };
  provider_status?: {provider?: string; model?: string; configured?: boolean; web_search_supported?: boolean; web_search_transport?: string};
  remote_consent?: {
    configured?: boolean;
    policy_version?: number;
    server_minted_receipts_required?: boolean;
    legacy_browser_receipts_authoritative?: boolean;
    grant_list_revoke_enabled?: boolean;
    purposes?: RemoteConsentPurpose[];
    policies?: RemoteConsentPolicy[];
  };
  visual_semantics?: {
    available?: boolean;
    provider_id?: string | null;
    processing_region?: string | null;
    sends_raw_media_remotely?: boolean;
  };
  teacher_authority?: {
    mode: "local_python" | "authenticated_apps_api";
    role_authorized: boolean;
    correct_mastery_updates_enabled: boolean;
    assurance:
      | "no_authenticated_teacher_identity"
      | "deployment_service_role_authorization_not_personal_signature";
    raw_identity_exposed: false;
    nonce_replay_policy?: "unavailable" | "append_only_permanent_tombstone_bounded_fail_closed";
    replay_store_max_bytes?: number;
    expired_nonce_tombstones_retained?: boolean;
  };
  interaction_contract?: Record<string, unknown>;
  skills?: Array<{
    skill_id: string;
    name: string;
    role: string;
    focus_dimension?: string;
    selection_rationale?: string;
    source?: Record<string, unknown>;
  }>;
  agent_runtime_policy?: {
    agent_loop_enabled?: boolean;
    maximum_agent_steps?: number;
    maximum_agent_tool_calls_per_step?: number;
    recoverable_context?: boolean;
    structured_tool_allowlist?: boolean;
  };
  default_goal?: Record<string, unknown>;
  default_student_profile?: Record<string, unknown>;
}

export interface TeacherAction {
  type?: string;
  message?: string;
  wait_for_student_before_next_action?: boolean;
  direct_answer_prohibited?: boolean;
  expected_signal?: string;
  [key: string]: unknown;
}

export interface AgentAction {
  action_id?: string;
  round?: number;
  type?: string;
  primary_skill?: {skill_id?: string; name?: string; role?: string; [key: string]: unknown};
  supporting_skills?: Array<{skill_id?: string; name?: string; role?: string; [key: string]: unknown}>;
  teacher_action?: TeacherAction;
  lesson_phase?: LessonProgress;
  selection_reason?: string;
  skill_switched?: boolean;
  [key: string]: unknown;
}

export interface TeacherHistoryEvent {
  round?: number;
  action?: AgentAction;
  learner_response?: string;
  learner_text?: string;
  structured_signal?: {label?: string; confidence?: number; [key: string]: unknown};
  [key: string]: unknown;
}

export interface TeacherSessionResponse {
  session_id: string;
  response_sha256: string;
  status?: SessionStatus;
  rounds_completed?: number;
  round?: number;
  goal?: Record<string, unknown>;
  student_state?: {
    knowledge_mastery?: Record<string, number>;
    misconceptions?: Array<Record<string, unknown>>;
    understanding_signal?: Record<string, unknown>;
    [key: string]: unknown;
  };
  next_action?: AgentAction;
  current_action?: AgentAction;
  lesson_progress?: LessonProgress | null;
  history?: TeacherHistoryEvent[];
  context_version: number;
  expected_question_id: string | null;
  profile_summary?: {
    profile_revision: string;
    display_name?: string;
    learner_level?: string;
    preferences?: string[];
    initial_mastery?: Record<string, number>;
    [key: string]: unknown;
  };
  setup_snapshot?: {
    goal?: {
      concept?: string;
      objective?: string;
      knowledge_components?: string[];
      [key: string]: unknown;
    };
    student_profile?: {
      profile_ref?: string;
      learner_level?: string;
      preferences?: string[];
      initial_mastery?: Record<string, number>;
      known_misconceptions?: Array<{
        tag?: string;
        description?: string;
        confidence?: number;
        [key: string]: unknown;
      }>;
      background_history?: unknown[];
      conversation_history?: unknown[];
      accessibility_needs?: string[];
      contains_direct_identity?: boolean;
      [key: string]: unknown;
    };
    [key: string]: unknown;
  };
  pending_skill_id?: string | null;
  control_notice?: string | null;
  teaching_resources?: TeachingResourceSummary[];
  agent_runtime?: Record<string, unknown>;
  claim_boundary?: Record<string, unknown>;
  [key: string]: unknown;
}

export type MetacognitionStrategyCode =
  | "retrieval"
  | "self_explanation"
  | "decomposition"
  | "worked_example"
  | "analogy"
  | "elimination"
  | "diagram"
  | "checking";

export interface LearningReviewDueItem {
  review_id: string;
  curriculum_namespace: string;
  knowledge_component_id: string;
  source_ref_sha256: string;
  due_at_utc: string;
  expected_version: number;
  claim_state: "unclaimed" | "expired";
}

export interface LearningReviewComponent {
  curriculum_namespace: string;
  knowledge_component_id: string;
  version: number;
  state: "scheduled" | "in_progress" | string;
  due_at_utc: string | null;
  interval_days: number | null;
  active_review_id: string | null;
  active_lease_id: string | null;
  lease_expires_at_utc: string | null;
  review_session_id?: string;
}

export interface LearningReviewDueList {
  schema: "teaching_skill_miner.learning_review_due_list.v1";
  session_id: string;
  reviews: LearningReviewDueItem[];
  active_reviews: LearningReviewComponent[];
  direct_client_outcome_updates_allowed: false;
}

export interface LearningReviewClaimResponse {
  schema: "teaching_skill_miner.learning_review_claim.v2";
  session_id: string;
  controller_session_id: string;
  review_id: string;
  applied: boolean;
  component: LearningReviewComponent;
  review_session: TeacherSessionResponse;
  direct_client_outcome_updates_allowed: false;
}

export interface LearningReviewReleaseResponse {
  schema: "teaching_skill_miner.learning_review_release.v1";
  session_id: string;
  review_id: string;
  applied: boolean;
  component: LearningReviewComponent;
}

export interface MetacognitionFeedback {
  authoritative_outcome: "correct" | "partial" | "incorrect";
  actual_score_percent?: number;
  calibration_classification: "overconfident" | "underconfident" | "aligned";
  strategy_codes?: MetacognitionStrategyCode[];
  message_zh: string;
  scoring_standard_changed: false;
  mastery_changed_by_metacognition: false;
  external_calibration_established: false;
}

export interface MetacognitionPredictionItem {
  prediction_event_id: string;
  occurred_at_utc: string;
  knowledge_component_id: string;
  assessment_kind: "verification" | "transfer" | "delayed_review";
  item_id: string;
  question_id: string;
  review_id: string | null;
  lease_id: string | null;
  learner_jol_percent: number;
  strategy_codes: MetacognitionStrategyCode[];
  status: "pending" | "paired";
  pairing_event_id?: string;
  feedback?: MetacognitionFeedback;
  outcome_accepted_from_client: false;
  mastery_changed: false;
}

export interface MetacognitionSessionProjection {
  schema: "teaching_skill_miner.metacognition_session_projection.v1";
  session_id: string;
  predictions: MetacognitionPredictionItem[];
  contains_learner_answer: false;
  outcome_accepted_from_client: false;
  mastery_changed: false;
  external_calibration_established: false;
}

export interface MetacognitionPredictionReceipt {
  schema: "teaching_skill_miner.metacognition_prediction_receipt.v1";
  session_id: string;
  prediction_event_id: string;
  applied: boolean;
  prompt_contract: {
    target: {knowledge_component_id: string; knowledge_component_label: string};
    attempt: {assessment_kind: "verification" | "transfer" | "delayed_review"};
    capture_order: "before_learner_answer";
  };
  confidence_source: "learner_self_report";
  assessment_confidence_used_as_learner_jol: false;
  outcome_accepted_from_client: false;
  mastery_changed: false;
  external_calibration_established: false;
}

export interface MetacognitionPairingReceipt {
  schema: "teaching_skill_miner.metacognition_pairing_receipt.v1";
  session_id: string;
  prediction_event_id: string;
  pairing_event_id: string;
  applied: boolean;
  feedback: MetacognitionFeedback;
  outcome_accepted_from_client: false;
  scoring_standard_changed: false;
  mastery_changed: false;
  external_calibration_established: false;
}

export type AdjudicationDecision = "approve" | "correct" | "abstain";

export interface AdjudicationReviewItem {
  schema: "teaching_skill_miner.teacher_agent_adjudication_item.v1";
  item_id: string;
  version: number;
  status: "pending" | "claimed" | "decided" | "cancelled";
  created_at: string;
  updated_at: string;
  source: {
    session_id: string;
    round_number: number;
    action_id: string;
    question_id: string;
    history_event_sha256: string;
  };
  target_kc_ids: string[];
  target_scope: "exact_one" | "bounded_set";
  original: {
    assessment_id: string;
    assessment_sha256: string;
    evidence_id: string;
    evidence_sha256: string;
    rubric_id: string;
    rubric_authority_sha256: string;
  };
  review_reason: string;
  lease?: {
    expires_at?: string;
    actor?: {identity?: string; authenticated?: boolean};
  } | null;
  decision?: {kind?: AdjudicationDecision; reason_code?: string} | null;
  cancellation?: {reason?: string; cancelled_at?: string} | null;
  version_sha256: string;
}

export interface AdjudicationReviewList {
  schema: "teaching_skill_miner.dashboard_adjudication_list.v1";
  session_id: string;
  items: AdjudicationReviewItem[];
  operator_identity: "local_operator_not_authenticated" | "authenticated_teacher_server_authorized";
  authenticated_teacher: boolean;
  public_items_contain_learner_text: false;
  correct_requires_authority_revalidation: true;
}

export interface AdjudicationCandidate {
  history_round: number;
  knowledge_component_id: string;
  knowledge_component_label?: string;
  question_id?: string;
  rubric_id?: string;
  evidence_locator_sha256: string;
}

export interface AdjudicationClaimResponse {
  schema: "teaching_skill_miner.dashboard_adjudication_claim.v1";
  session_id: string;
  item: AdjudicationReviewItem;
  claim_token: string;
  operator_identity: "local_operator_not_authenticated" | "authenticated_teacher_server_authorized";
  authenticated_teacher: boolean;
}

export interface AdjudicationDecisionResponse {
  schema: "teaching_skill_miner.dashboard_adjudication_decision.v1";
  session_id: string;
  session_ref: {
    session_id: string;
    expected_round: number;
    expected_question_id: string | null;
    expected_context_version: number;
    profile_revision: string;
  };
  item: AdjudicationReviewItem;
  model_effect: {
    status: "approved_no_change" | "pending_authority_revalidation" | "applied_supersede_replay" | "already_applied_no_change" | "pending_external_ledger";
    pending?: {reason?: string; model_mutated?: false; mastery_update_authorized?: false} | null;
    revision_receipt?: Record<string, unknown> | null;
  };
  operator_identity: "local_operator_not_authenticated" | "authenticated_teacher_server_authorized";
  authenticated_teacher: boolean;
  correct_requires_authority_revalidation: boolean;
}

export interface TeacherCommandResponse extends TeacherSessionResponse {}

export interface TeacherAttachmentResponse {
  session_id: string;
  context_version: number;
  expected_question_id: string;
  profile_revision: string;
  attachment: {
    attachment_id: string;
    recognized_text?: string;
    needs_student_confirmation?: boolean;
    [key: string]: unknown;
  };
}

export interface TeacherResourceResponse extends Partial<TeacherSessionResponse> {
  staged?: boolean;
  resource: TeachingResourceSummary;
  session_use?: {
    status: "eligible_untrusted_context" | "blocked_pending_confirmation" | "blocked_abstained";
    decision?: string | null;
    reason?: string | null;
    grading_evidence_allowed: false;
    mastery_evidence_allowed: false;
  };
}

export interface TaskEvent {
  id: string;
  type: "status" | "message" | "tool" | "state" | "error";
  sessionId: string;
  payload: Record<string, unknown>;
  occurredAt: string;
}
