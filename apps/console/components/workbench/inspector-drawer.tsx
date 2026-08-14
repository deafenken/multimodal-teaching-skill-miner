"use client";

import React from "react";
import {
  Bot,
  BookOpen,
  BrainCircuit,
  ChevronLeft,
  Gauge,
  Home,
  Palette,
  Scale,
  Settings,
  ShieldAlert,
  ShieldCheck,
  UserRound,
  X
} from "lucide-react";

import {Button} from "@/components/ui/button";
import {Separator} from "@/components/ui/separator";
import {Switch} from "@/components/ui/switch";
import {cn} from "@/lib/cn";
import {teacherAuthorityUiState} from "@/lib/teacher-authority";
import {
  acknowledgeSafeguardingCase,
  acknowledgeSafeguardingDelivery,
  claimDueLearningReview,
  closeSafeguardingCase,
  claimAdjudicationReview,
  decideAdjudicationReview,
  enqueueAdjudicationReview,
  grantRemoteConsent,
  listAdjudicationCandidates,
  listAdjudicationReviews,
  listDueLearningReviews,
  listMetacognitionPredictions,
  listSafeguardingCases,
  listRemoteConsents,
  pairMetacognitionPrediction,
  recordMetacognitionPrediction,
  releaseLearningReview,
  reviewTeachingResource,
  resumeTeachingSession,
  revokeRemoteConsent,
} from "@/lib/api";
import {
  validSafeguardingRouteLocator,
  type SafeguardingStaffCase,
} from "@/lib/safeguarding-staff";
import {
  buildResourceReviewRequest,
  newResourceReviewIdempotencyKey,
  RESOURCE_REVIEW_MAX_NOTE_CHARS,
  RESOURCE_REVIEW_MAX_TEXT_CHARS,
  resourceReviewAvailability,
  resourceReviewRequirements,
  safeReviewedResource,
  type ResourceReviewAttestationKey,
  type ResourceReviewDraft,
} from "@/lib/resource-review";
import type {AdjudicationCandidate, AdjudicationReviewItem, BootstrapPayload, LearningReviewComponent, LearningReviewDueItem, LessonProgress, MetacognitionPredictionItem, MetacognitionStrategyCode, RemoteConsentPurpose, RemoteConsentReceipt, TeacherSessionResponse, TeachingResourceSummary} from "@/lib/types";

type InspectorTab = "profile" | "state" | "resources" | "method" | "review" | "consent" | "safeguarding" | "appearance" | "runtime";

const consentPurposeLabels: Record<RemoteConsentPurpose, string> = {
  remote_chat: "Chat",
  remote_teaching: "Teach",
  remote_syllabus_generation: "大纲生成",
  public_web_search: "联网搜索",
  remote_visual_analysis: "远程视觉分析",
};

function consentReceiptIsCurrent(
  receipt: RemoteConsentReceipt,
  policies: NonNullable<BootstrapPayload["remote_consent"]>["policies"],
) {
  return receipt.status === "active"
    && Date.parse(receipt.expires_at_utc) > Date.now()
    && Boolean(policies?.some((policy) => policy.purpose === receipt.purpose
      && receipt.provider_id === policy.provider_id
      && receipt.processing_region === policy.processing_region
      && receipt.provider_retention_days === policy.provider_retention_days
      && receipt.provider_policy_sha256 === policy.provider_policy_sha256
      && receipt.subject_policy_sha256 === policy.subject_policy_sha256
      && policy.data_categories.every((category) => receipt.data_categories.includes(category))));
}

const masteryLabels: Record<string, string> = {
  prerequisite: "前置知识",
  conceptual: "概念理解",
  procedural: "操作过程",
  transfer: "迁移能力",
};

const metacognitionStrategyLabels: Record<MetacognitionStrategyCode, string> = {
  retrieval: "闭卷回想",
  self_explanation: "自我解释",
  decomposition: "拆分问题",
  worked_example: "参照例题",
  analogy: "类比迁移",
  elimination: "排除检验",
  diagram: "画图建模",
  checking: "二次检查",
};

const metacognitionStrategies = Object.keys(metacognitionStrategyLabels) as MetacognitionStrategyCode[];

function visibleLessonProgress(progress?: LessonProgress | null) {
  const label = progress?.phase_label?.trim();
  const index = Number(progress?.phase_index);
  if (!label || progress?.phase_count !== 6 || !Number.isInteger(index) || index < 1 || index > 6) return null;
  return {label, index, currentConcept: progress.current_concept?.trim()};
}

function SegmentedControl({value, options, onChange, label, optionLabels = {}}: {value: string; options: string[]; onChange: (value: string) => void; label: string; optionLabels?: Record<string, string>}) {
  return (
    <div role="group" aria-label={label} className="flex w-full rounded-xl bg-[var(--inspector-control)] p-1 min-[561px]:w-auto">
      {options.map((option) => (
        <button
          key={option}
          type="button"
          aria-pressed={value === option}
          onClick={() => onChange(option)}
          className={cn("min-w-0 flex-1 rounded-lg px-4 py-2 text-sm text-[var(--inspector-muted)] min-[561px]:min-w-20", value === option && "bg-[var(--inspector-card)] text-[var(--inspector-text)] shadow-sm")}
        >
          {optionLabels[option] ?? option}
        </button>
      ))}
    </div>
  );
}

function SettingRow({title, description, children, compact = false}: {title: string; description: string; children: React.ReactNode; compact?: boolean}) {
  return (
    <div className={cn("grid grid-cols-1 items-center gap-4 min-[561px]:grid-cols-[minmax(0,1fr)_auto] min-[561px]:gap-6", compact ? "py-4" : "py-7")}>
      <div>
        <h3 className="text-base font-semibold text-[var(--inspector-text)]">{title}</h3>
        <p className="mt-1 text-sm text-[var(--inspector-muted)]">{description}</p>
      </div>
      <div className="w-full min-[561px]:w-auto min-[561px]:justify-self-end">{children}</div>
    </div>
  );
}

const resourceReviewAttestations: Array<{key: ResourceReviewAttestationKey; label: string}> = [
  {key: "compared_with_original_source", label: "我已逐项对照原始资料，而不是只看自动提取结果"},
  {key: "uncertainties_removed_or_explicit", label: "不确定内容已删除，或已在正文中明确标注"},
  {key: "not_an_answer_key", label: "这段文字不是答案键，也不授权判定学生回答"},
  {key: "context_only", label: "这段文字只作为不受信任的教学上下文"},
];

function blankResourceReviewDraft(): ResourceReviewDraft {
  return {
    reviewedText: "",
    excludedLayerIds: [],
    attestations: {
      compared_with_original_source: false,
      uncertainties_removed_or_explicit: false,
      not_an_answer_key: false,
      context_only: false,
    },
    reviewNote: "",
  };
}

function TeachingResourceCard({resource, teacherAuthority, onReviewed}: {
  resource: TeachingResourceSummary;
  teacherAuthority?: BootstrapPayload["teacher_authority"];
  onReviewed: (resource: TeachingResourceSummary) => void;
}) {
  const [draft, setDraft] = React.useState<ResourceReviewDraft>(blankResourceReviewDraft);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [notice, setNotice] = React.useState<string | null>(null);
  const attemptKeyRef = React.useRef<string | null>(null);
  const availability = resourceReviewAvailability(teacherAuthority, resource);
  const requirements = resourceReviewRequirements(resource);
  const contract = resource.evidence_contract;
  const conflicts = requirements?.conflicts ?? contract?.conflicts ?? [];
  const layers = requirements?.layers ?? contract?.layers ?? [];
  const decision = contract?.decision ?? "unknown";
  const reviewed = resource.resource_review?.reviewed === true;

  const updateDraft = (update: (current: ResourceReviewDraft) => ResourceReviewDraft) => {
    attemptKeyRef.current = null;
    setDraft(update);
    setError(null);
    setNotice(null);
  };

  const submitReview = async () => {
    if (busy || !availability.enabled) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      attemptKeyRef.current ??= newResourceReviewIdempotencyKey();
      const request = buildResourceReviewRequest(resource, draft, attemptKeyRef.current);
      const response = await reviewTeachingResource(request);
      const safe = safeReviewedResource(response, resource);
      onReviewed(safe);
      setDraft(blankResourceReviewDraft());
      attemptKeyRef.current = null;
      setNotice(`复核版本 v${safe.resource_review?.review_version ?? "—"} 已保存；仅用于教学上下文。`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "资源复核失败");
    } finally {
      setBusy(false);
    }
  };

  return (
    <article className="rounded-xl border border-[var(--inspector-border)] bg-[var(--inspector-card)] p-4 shadow-sm">
      <div className="flex items-start gap-3">
        <span className="grid size-9 shrink-0 place-items-center rounded-lg bg-[var(--app-success-soft)] text-[var(--app-success)]"><BookOpen className="size-4" /></span>
        <div className="min-w-0 flex-1">
          <strong className="block truncate" title={resource.display_name}>{resource.display_name}</strong>
          <p className="mt-1 text-xs text-[var(--inspector-muted)]">{resource.resource_type ?? "document"}{resource.page_count ? ` · ${resource.page_count} 页` : ""} · {resource.extracted_char_count ?? 0} 字</p>
        </div>
        {reviewed && <span className="shrink-0 rounded-full border border-[var(--app-success)] bg-[var(--app-success-soft)] px-2 py-1 text-[10px] text-[var(--app-success)]">已复核 v{resource.resource_review?.review_version}</span>}
      </div>
      <div className="mt-3 flex flex-wrap gap-2 text-[10px]">
        <span className="rounded-full border border-[var(--inspector-border)] px-2 py-1">本机提取</span>
        <span className="rounded-full border border-[var(--inspector-border)] px-2 py-1">原始媒体未发送</span>
        <span className="rounded-full border border-[var(--inspector-border)] px-2 py-1">{layers.length} 个证据层</span>
        <span className={cn("rounded-full border px-2 py-1", conflicts.length ? "border-[var(--app-warning)] bg-[var(--app-warning-soft)] text-[var(--app-warning)]" : "border-[var(--inspector-border)]")}>{conflicts.length} 个冲突</span>
        {resource.truncated && <span className="rounded-full border border-[var(--app-warning)] bg-[var(--app-warning-soft)] px-2 py-1 text-[var(--app-warning)]">超长内容已截断</span>}
        {resource.needs_review && !reviewed && <span className="rounded-full border border-[var(--app-warning)] bg-[var(--app-warning-soft)] px-2 py-1 text-[var(--app-warning)]">需要人工核对</span>}
      </div>

      <dl className="mt-3 grid gap-1 rounded-lg bg-[var(--inspector-control)] p-3 text-xs leading-5">
        <div><dt className="inline font-medium">使用决定：</dt><dd className="inline text-[var(--inspector-muted)]">{decision}</dd></div>
        <div><dt className="inline font-medium">安全边界：</dt><dd className="inline text-[var(--inspector-muted)]">不受信任的教学上下文；不是答案键、评分或掌握度证据</dd></div>
      </dl>

      {conflicts.length > 0 && (
        <details className="mt-3 rounded-lg border border-[var(--app-warning)] bg-[var(--app-warning-soft)] p-3" open={!reviewed}>
          <summary className="cursor-pointer text-xs font-semibold text-[var(--app-warning)]">待处理冲突（{conflicts.length}）</summary>
          <ul className="mt-2 grid gap-2 text-xs leading-5 text-[var(--inspector-muted)]">
            {conflicts.map((conflict) => <li key={conflict.conflict_id}><strong className="text-[var(--inspector-text)]">{conflict.kind}</strong>：{conflict.description}</li>)}
          </ul>
          <p className="mt-2 text-[10px] leading-4 text-[var(--inspector-muted)]">提交时会把上面全部冲突 ID 明确列为已处理；浏览器不会省略或猜测隐藏冲突。</p>
        </details>
      )}

      <details className="mt-3 rounded-lg border border-[var(--inspector-border)] p-3">
        <summary className="cursor-pointer text-xs font-semibold">证据层（{layers.length}）</summary>
        {!layers.length ? <p className="mt-2 text-xs text-[var(--inspector-muted)]">没有可复核的证据层。</p> : (
          <div className="mt-2 grid gap-2">
            {layers.map((layer) => {
              const checked = draft.excludedLayerIds.includes(layer.layer_id);
              return (
                <label key={layer.layer_id} className="flex items-start gap-2 rounded-lg bg-[var(--inspector-control)] p-2 text-xs">
                  <input
                    type="checkbox"
                    className="mt-0.5 accent-[var(--app-accent)]"
                    checked={checked}
                    disabled={busy || !availability.enabled}
                    onChange={(event) => updateDraft((current) => ({
                      ...current,
                      excludedLayerIds: event.target.checked
                        ? [...current.excludedLayerIds, layer.layer_id]
                        : current.excludedLayerIds.filter((id) => id !== layer.layer_id),
                    }))}
                  />
                  <span className="min-w-0"><strong>{layer.kind}</strong><span className="ml-1 text-[var(--inspector-muted)]">{layer.status} · {layer.evidence_locator ?? layer.layer_id}</span><span className="mt-0.5 block text-[10px] text-[var(--inspector-muted)]">勾选表示从复核投影中排除此层</span></span>
                </label>
              );
            })}
          </div>
        )}
      </details>

      <details className="mt-3 rounded-lg border border-[var(--inspector-border)] p-3" open={!reviewed && Boolean(resource.needs_review || resource.requires_confirmation)}>
        <summary className="cursor-pointer text-xs font-semibold">{reviewed ? "提交新修订" : "认证教师复核"}</summary>
        <div className="mt-3 grid gap-3">
          <p className={cn("rounded-lg p-3 text-xs leading-5", availability.enabled ? "bg-[var(--app-success-soft)] text-[var(--app-success)]" : "bg-[var(--app-warning-soft)] text-[var(--app-warning)]")} role="note">{availability.reason}</p>
          <label className="grid gap-1 text-xs text-[var(--inspector-muted)]">
            与原始资料核对后的文本
            <textarea
              rows={7}
              maxLength={RESOURCE_REVIEW_MAX_TEXT_CHARS}
              value={draft.reviewedText}
              disabled={busy || !availability.enabled}
              onChange={(event) => updateDraft((current) => ({...current, reviewedText: event.target.value}))}
              placeholder="只填写已对照原文件确认的教学内容；不要填写答案键或评分结论。"
              className="resize-y rounded-lg border border-[var(--inspector-border)] bg-[var(--inspector-control)] px-3 py-2 leading-5 text-[var(--inspector-text)] outline-none focus:border-[var(--app-accent)] disabled:cursor-not-allowed disabled:opacity-60"
            />
            <span className="justify-self-end text-[10px]">{draft.reviewedText.length.toLocaleString()} / {RESOURCE_REVIEW_MAX_TEXT_CHARS.toLocaleString()}</span>
          </label>
          <label className="grid gap-1 text-xs text-[var(--inspector-muted)]">
            复核备注（可选）
            <textarea
              rows={2}
              maxLength={RESOURCE_REVIEW_MAX_NOTE_CHARS}
              value={draft.reviewNote}
              disabled={busy || !availability.enabled}
              onChange={(event) => updateDraft((current) => ({...current, reviewNote: event.target.value}))}
              className="resize-y rounded-lg border border-[var(--inspector-border)] bg-[var(--inspector-control)] px-3 py-2 leading-5 text-[var(--inspector-text)] outline-none focus:border-[var(--app-accent)] disabled:cursor-not-allowed disabled:opacity-60"
            />
          </label>
          <fieldset className="grid gap-2" disabled={busy || !availability.enabled}>
            <legend className="mb-1 text-xs font-semibold">四项教师声明（全部必选）</legend>
            {resourceReviewAttestations.map((attestation) => (
              <label key={attestation.key} className="flex items-start gap-2 text-xs leading-5 text-[var(--inspector-muted)]">
                <input
                  type="checkbox"
                  className="mt-1 accent-[var(--app-accent)]"
                  checked={draft.attestations[attestation.key]}
                  onChange={(event) => updateDraft((current) => ({...current, attestations: {...current.attestations, [attestation.key]: event.target.checked}}))}
                />
                <span>{attestation.label}</span>
              </label>
            ))}
          </fieldset>
          <p className="text-[10px] leading-4 text-[var(--inspector-muted)]">仅发送这段有界文字、冲突/层 ID、声明与备注。原文件、图片、音视频不会随复核请求发送。</p>
          {error && <p className="rounded-lg border border-red-400/40 bg-red-500/10 p-3 text-xs leading-5 text-red-300" role="alert">{error}</p>}
          {notice && <p className="rounded-lg border border-[var(--app-success)]/40 bg-[var(--app-success-soft)] p-3 text-xs leading-5 text-[var(--app-success)]" role="status">{notice}</p>}
          <Button className="w-full" size="sm" disabled={busy || !availability.enabled} onClick={() => void submitReview()}>{busy ? "正在安全提交…" : reviewed ? "保存新修订" : "提交认证复核"}</Button>
        </div>
      </details>
    </article>
  );
}

export function InspectorDrawer({open, onClose, session, teachingResources: teachingResourcesProp, providerStatus, remoteConsent, visualSemantics, teacherAuthority, settingsRequest = 0, resourcesRequest = 0, webSearchEnabled = true, onWebSearchEnabledChange = () => undefined, visualAnalysisEnabled = false, onVisualAnalysisEnabledChange = () => undefined, onSessionChange = () => undefined, onResourceReviewed = () => undefined}: {
  open: boolean;
  onClose: () => void;
  session?: TeacherSessionResponse | null;
  teachingResources?: TeachingResourceSummary[];
  providerStatus?: {provider?: string; model?: string; configured?: boolean; web_search_supported?: boolean};
  remoteConsent?: BootstrapPayload["remote_consent"];
  visualSemantics?: BootstrapPayload["visual_semantics"];
  teacherAuthority?: BootstrapPayload["teacher_authority"];
  settingsRequest?: number;
  resourcesRequest?: number;
  webSearchEnabled?: boolean;
  onWebSearchEnabledChange?: (enabled: boolean) => void;
  visualAnalysisEnabled?: boolean;
  onVisualAnalysisEnabledChange?: (enabled: boolean) => void;
  onSessionChange?: (session: TeacherSessionResponse) => void;
  onResourceReviewed?: (resource: TeachingResourceSummary) => void;
}) {
  const [tab, setTab] = React.useState<InspectorTab>("profile");
  const [theme, setTheme] = React.useState("Dark");
  const [density, setDensity] = React.useState("Default");
  const [reduceMotion, setReduceMotion] = React.useState(false);
  const [preferencesReady, setPreferencesReady] = React.useState(false);
  const [isMobile, setIsMobile] = React.useState(false);
  const panelRef = React.useRef<HTMLElement>(null);
  const closeButtonRef = React.useRef<HTMLButtonElement>(null);
  const [adjudicationItems, setAdjudicationItems] = React.useState<AdjudicationReviewItem[]>([]);
  const [adjudicationCandidates, setAdjudicationCandidates] = React.useState<AdjudicationCandidate[]>([]);
  const [adjudicationTokens, setAdjudicationTokens] = React.useState<Record<string, string>>({});
  const [adjudicationSignals, setAdjudicationSignals] = React.useState<Record<string, "correct" | "partial" | "misconception">>({});
  const [adjudicationBusy, setAdjudicationBusy] = React.useState<string | null>(null);
  const [adjudicationError, setAdjudicationError] = React.useState<string | null>(null);
  const [adjudicationNotice, setAdjudicationNotice] = React.useState<string | null>(null);
  const [consentReceipts, setConsentReceipts] = React.useState<RemoteConsentReceipt[]>([]);
  const [consentBusy, setConsentBusy] = React.useState<string | null>(null);
  const [consentError, setConsentError] = React.useState<string | null>(null);
  const [consentNotice, setConsentNotice] = React.useState<string | null>(null);
  const [learnerJol, setLearnerJol] = React.useState(50);
  const [metacognitionStrategiesSelected, setMetacognitionStrategiesSelected] = React.useState<MetacognitionStrategyCode[]>([]);
  const [metacognitionItems, setMetacognitionItems] = React.useState<MetacognitionPredictionItem[]>([]);
  const [dueLearningReviews, setDueLearningReviews] = React.useState<LearningReviewDueItem[]>([]);
  const [activeLearningReviews, setActiveLearningReviews] = React.useState<LearningReviewComponent[]>([]);
  const [learningBusy, setLearningBusy] = React.useState<string | null>(null);
  const [learningError, setLearningError] = React.useState<string | null>(null);
  const [learningNotice, setLearningNotice] = React.useState<string | null>(null);
  const [safeguardingRouteLocator, setSafeguardingRouteLocator] = React.useState("");
  const [safeguardingCases, setSafeguardingCases] = React.useState<SafeguardingStaffCase[]>([]);
  const [safeguardingBusy, setSafeguardingBusy] = React.useState<string | null>(null);
  const [safeguardingError, setSafeguardingError] = React.useState<string | null>(null);
  const [safeguardingNotice, setSafeguardingNotice] = React.useState<string | null>(null);
  const authorityUi = teacherAuthorityUiState(teacherAuthority);

  const refreshConsents = React.useCallback(async () => {
    if (!remoteConsent?.configured) {
      setConsentReceipts([]);
      setConsentError("后端没有配置服务器同意凭据存储；所有远程效果都会被拒绝。");
      return;
    }
    setConsentError(null);
    try {
      const response = await listRemoteConsents();
      setConsentReceipts(response.receipts);
    } catch (caught) {
      setConsentError(caught instanceof Error ? caught.message : "同意收据读取失败");
    }
  }, [remoteConsent?.configured]);

  const grantConsent = React.useCallback(async (purpose: RemoteConsentPurpose) => {
    setConsentBusy(purpose);
    setConsentError(null);
    setConsentNotice(null);
    try {
      await grantRemoteConsent({
        purpose,
        validity_days: 30,
      });
      await refreshConsents();
      setConsentNotice(`${consentPurposeLabels[purpose]} 已签发 30 天服务器收据；可随时撤销。`);
    } catch (caught) {
      setConsentError(caught instanceof Error ? caught.message : "同意收据签发失败");
    } finally {
      setConsentBusy(null);
    }
  }, [refreshConsents]);

  const revokeConsent = React.useCallback(async (receipt: RemoteConsentReceipt) => {
    setConsentBusy(receipt.consent_id);
    setConsentError(null);
    setConsentNotice(null);
    try {
      await revokeRemoteConsent(receipt.consent_id);
      await refreshConsents();
      setConsentNotice(`${consentPurposeLabels[receipt.purpose]} 已撤销；下一次远程效果会立即失败关闭。`);
    } catch (caught) {
      setConsentError(caught instanceof Error ? caught.message : "同意收据撤销失败");
    } finally {
      setConsentBusy(null);
    }
  }, [refreshConsents]);

  const refreshSafeguardingCases = React.useCallback(async () => {
    if (!validSafeguardingRouteLocator(safeguardingRouteLocator)) {
      setSafeguardingError("请输入中央安全保障系统提供的有效路由凭据；浏览器角色不会被采用。");
      return;
    }
    setSafeguardingBusy("list");
    setSafeguardingError(null);
    setSafeguardingNotice(null);
    try {
      const response = await listSafeguardingCases(safeguardingRouteLocator);
      setSafeguardingCases(response.cases);
      setSafeguardingNotice(`已读取 ${response.cases.length} 条仅含哈希和状态的案例投影。`);
    } catch (caught) {
      setSafeguardingCases([]);
      setSafeguardingError(caught instanceof Error ? caught.message : "安全保障案例读取失败");
    } finally {
      setSafeguardingBusy(null);
    }
  }, [safeguardingRouteLocator]);

  const mutateSafeguardingCase = React.useCallback(async (
    item: SafeguardingStaffCase,
    operation: "case_acknowledge" | "delivery_acknowledge" | "case_close",
  ) => {
    if (!validSafeguardingRouteLocator(safeguardingRouteLocator)) {
      setSafeguardingError("安全保障路由凭据无效；变更没有发送。");
      return;
    }
    setSafeguardingBusy(`${operation}:${item.case_id}`);
    setSafeguardingError(null);
    setSafeguardingNotice(null);
    try {
      const response = operation === "case_acknowledge"
        ? await acknowledgeSafeguardingCase(safeguardingRouteLocator, item.case_id, item.version)
        : operation === "delivery_acknowledge"
          ? await acknowledgeSafeguardingDelivery(safeguardingRouteLocator, item.case_id, item.version)
          : await closeSafeguardingCase(safeguardingRouteLocator, item.case_id, item.version);
      setSafeguardingCases((current) => current.map((candidate) => (
        candidate.case_id === response.case.case_id ? response.case : candidate
      )));
      setSafeguardingNotice(
        operation === "case_acknowledge"
          ? "案例已确认接手；操作使用服务器当前 safeguarding 角色和版本 CAS。"
          : operation === "delivery_acknowledge"
            ? "中央队列接收已确认；这不会自动关闭案例。"
            : "案例已关闭；服务端已验证接手和投递状态。",
      );
    } catch (caught) {
      setSafeguardingError(caught instanceof Error ? caught.message : "安全保障案例变更失败");
    } finally {
      setSafeguardingBusy(null);
    }
  }, [safeguardingRouteLocator]);

  const refreshLongTermLearning = React.useCallback(async (autoPair = true) => {
    if (!session) {
      setDueLearningReviews([]);
      setActiveLearningReviews([]);
      setMetacognitionItems([]);
      return;
    }
    setLearningError(null);
    const [reviewResult, metacognitionResult] = await Promise.allSettled([
      listDueLearningReviews(session),
      listMetacognitionPredictions(session),
    ]);
    const errors: string[] = [];
    if (reviewResult.status === "fulfilled") {
      setDueLearningReviews(reviewResult.value.reviews);
      setActiveLearningReviews(reviewResult.value.active_reviews);
    } else {
      errors.push(reviewResult.reason instanceof Error ? reviewResult.reason.message : "复习队列读取失败");
    }
    if (metacognitionResult.status === "fulfilled") {
      let projection = metacognitionResult.value;
      if (autoPair) {
        let pairedAny = false;
        for (const item of projection.predictions) {
          if (item.status !== "pending") continue;
          try {
            await pairMetacognitionPrediction(session, item.prediction_event_id);
            pairedAny = true;
          } catch {
            // A prediction legitimately remains pending until its bound
            // authoritative turn receipt exists. Never synthesize an outcome.
          }
        }
        if (pairedAny) {
          projection = await listMetacognitionPredictions(session);
          setLearningNotice("已用权威教学回合收据自动配对预测；评分标准和掌握度未改变。");
        }
      }
      setMetacognitionItems(projection.predictions);
    } else {
      errors.push(metacognitionResult.reason instanceof Error ? metacognitionResult.reason.message : "自评记录读取失败");
    }
    if (errors.length) setLearningError(errors.join("；"));
  }, [session]);

  const submitMetacognitionPrediction = React.useCallback(async () => {
    if (!session) return;
    if (metacognitionStrategiesSelected.length < 1 || metacognitionStrategiesSelected.length > 3) {
      setLearningError("请选择 1–3 个实际准备使用的策略。");
      return;
    }
    setLearningBusy("prediction");
    setLearningError(null);
    setLearningNotice(null);
    try {
      await recordMetacognitionPrediction(session, {
        learnerJolPercent: learnerJol,
        strategyCodes: metacognitionStrategiesSelected,
      });
      setLearningNotice("答题前自评已持久化。它不是答案、评分或掌握度证据。");
      await refreshLongTermLearning(false);
    } catch (caught) {
      setLearningError(caught instanceof Error ? caught.message : "答题前自评记录失败");
    } finally {
      setLearningBusy(null);
    }
  }, [learnerJol, metacognitionStrategiesSelected, refreshLongTermLearning, session]);

  const explicitlyPairPrediction = React.useCallback(async (predictionEventId: string) => {
    if (!session) return;
    setLearningBusy(predictionEventId);
    setLearningError(null);
    try {
      await pairMetacognitionPrediction(session, predictionEventId);
      await refreshLongTermLearning(false);
      setLearningNotice("权威回合收据已配对；客户端没有提交结果或掌握度。");
    } catch (caught) {
      setLearningError(caught instanceof Error ? caught.message : "尚无可配对的权威回合收据");
    } finally {
      setLearningBusy(null);
    }
  }, [refreshLongTermLearning, session]);

  const claimLearningReview = React.useCallback(async (review: LearningReviewDueItem) => {
    if (!session) return;
    setLearningBusy(review.review_id);
    setLearningError(null);
    try {
      const claimed = await claimDueLearningReview(session, review.review_id, review.expected_version);
      setDueLearningReviews((current) => current.filter((item) => item.review_id !== review.review_id));
      setActiveLearningReviews([claimed.component]);
      setLearningNotice("服务端已打开到期复习题；答案将由权威教学回合收据判定。");
      onSessionChange(claimed.review_session);
      onClose();
    } catch (caught) {
      setLearningError(caught instanceof Error ? caught.message : "复习领取失败");
    } finally {
      setLearningBusy(null);
    }
  }, [onClose, onSessionChange, session]);

  const releaseOneLearningReview = React.useCallback(async (
    targetSession: TeacherSessionResponse,
    review: LearningReviewComponent,
    updateView = true,
  ) => {
    if (!review.active_review_id || !review.active_lease_id) return;
    await releaseLearningReview(targetSession, {
      reviewId: review.active_review_id,
      leaseId: review.active_lease_id,
      expectedVersion: review.version,
    });
    if (updateView) {
      setActiveLearningReviews((current) => current.filter((item) => item.active_lease_id !== review.active_lease_id));
      setLearningNotice("复习租约已安全释放；未上传客户端结果。");
    }
  }, []);

  const releaseLearningReviewFromView = React.useCallback(async (review: LearningReviewComponent) => {
    if (!session) return;
    setLearningBusy(review.active_review_id ?? "release");
    setLearningError(null);
    try {
      await releaseOneLearningReview(session, review);
    } catch (caught) {
      setLearningError(caught instanceof Error ? caught.message : "复习租约释放失败");
    } finally {
      setLearningBusy(null);
    }
  }, [releaseOneLearningReview, session]);

  const continueLearningReview = React.useCallback(async (review: LearningReviewComponent) => {
    if (!review.review_session_id) return;
    setLearningBusy(review.active_review_id ?? review.review_session_id);
    setLearningError(null);
    try {
      const reviewSession = await resumeTeachingSession(review.review_session_id);
      onSessionChange(reviewSession);
      setLearningNotice("已从服务端唯一租约绑定恢复复习会话；未使用浏览器保存的题目或答案。");
      onClose();
    } catch (caught) {
      setLearningError(caught instanceof Error ? caught.message : "复习会话恢复失败");
    } finally {
      setLearningBusy(null);
    }
  }, [onClose, onSessionChange]);

  const refreshAdjudications = React.useCallback(async () => {
    if (!session) {
      setAdjudicationItems([]);
      return;
    }
    setAdjudicationError(null);
    try {
      const [response, candidateResponse] = await Promise.all([
        listAdjudicationReviews(session),
        listAdjudicationCandidates(session),
      ]);
      setAdjudicationItems(response.items);
      setAdjudicationCandidates(candidateResponse.candidates);
    } catch (caught) {
      setAdjudicationError(caught instanceof Error ? caught.message : "复核队列读取失败");
    }
  }, [session]);

  const enqueueReview = React.useCallback(async (candidate: AdjudicationCandidate) => {
    if (!session) return;
    const busyId = candidate.evidence_locator_sha256;
    setAdjudicationBusy(busyId);
    setAdjudicationError(null);
    try {
      await enqueueAdjudicationReview(session, {
        historyRound: candidate.history_round,
        knowledgeComponentId: candidate.knowledge_component_id,
        reviewReason: "manual_quality_review",
      });
      await refreshAdjudications();
      setAdjudicationNotice("已由服务器重新解析并封印该历史证据；学生原文未进入复核 API。 ");
    } catch (caught) {
      setAdjudicationError(caught instanceof Error ? caught.message : "加入复核队列失败");
    } finally {
      setAdjudicationBusy(null);
    }
  }, [refreshAdjudications, session]);

  React.useEffect(() => {
    setAdjudicationTokens({});
    setAdjudicationNotice(null);
    if (open && tab === "review") void refreshAdjudications();
  }, [open, refreshAdjudications, session?.session_id, tab]);

  React.useEffect(() => {
    if (open && tab === "state") void refreshLongTermLearning(true);
  }, [open, refreshLongTermLearning, session?.response_sha256, tab]);

  React.useEffect(() => {
    if (open && tab === "consent") void refreshConsents();
  }, [open, refreshConsents, tab]);

  React.useEffect(() => {
    if (open) return;
    // The opaque locator is a routing capability. It is intentionally never
    // persisted and is discarded as soon as the inspector closes.
    setSafeguardingRouteLocator("");
    setSafeguardingCases([]);
    setSafeguardingError(null);
    setSafeguardingNotice(null);
  }, [open]);

  const claimReview = React.useCallback(async (item: AdjudicationReviewItem) => {
    if (!session) return;
    if (!authorityUi.actionsAllowed) {
      setAdjudicationError("当前 bootstrap 未声明可用的本机边界或已认证教师角色；裁决失败关闭。");
      return;
    }
    setAdjudicationBusy(item.item_id);
    setAdjudicationError(null);
    try {
      const response = await claimAdjudicationReview(session, item.item_id, item.version);
      setAdjudicationTokens((current) => ({...current, [item.item_id]: response.claim_token}));
      setAdjudicationItems((current) => current.map((candidate) => candidate.item_id === item.item_id ? response.item : candidate));
      setAdjudicationNotice(authorityUi.authenticatedTeacher
        ? "已取得认证教师复核租约。授权由部署服务按当前会话角色签发，不是个人不可否认签名。"
        : "已取得 15 分钟本地复核租约。此操作者不是已认证教师。");
    } catch (caught) {
      setAdjudicationError(caught instanceof Error ? caught.message : "复核领取失败");
    } finally {
      setAdjudicationBusy(null);
    }
  }, [authorityUi.actionsAllowed, authorityUi.authenticatedTeacher, session]);

  const decideReview = React.useCallback(async (item: AdjudicationReviewItem, decision: "approve" | "correct" | "abstain") => {
    if (!session) return;
    if (!authorityUi.actionsAllowed) {
      setAdjudicationError("当前身份没有教师裁决权限；服务器不会接收浏览器角色或 actor 字段。");
      return;
    }
    const claimToken = adjudicationTokens[item.item_id];
    if (!claimToken) {
      setAdjudicationError("请先领取这条复核项；claim token 只保存在当前页面内存中。");
      return;
    }
    setAdjudicationBusy(item.item_id);
    setAdjudicationError(null);
    try {
      const signal = adjudicationSignals[item.item_id] ?? "partial";
      const response = await decideAdjudicationReview(session, {
        itemId: item.item_id,
        expectedVersion: item.version,
        claimToken,
        decision,
        reasonCode: decision === "approve" ? "assessment_confirmed" : decision === "correct" ? "signal_misclassified" : "insufficient_evidence",
        ...(decision === "correct" ? {correction: {signal}} : {}),
      });
      setAdjudicationTokens((current) => {
        const next = {...current};
        delete next[item.item_id];
        return next;
      });
      setAdjudicationItems((current) => current.map((candidate) => candidate.item_id === item.item_id ? response.item : candidate));
      const refreshedSession = await resumeTeachingSession(response.session_ref.session_id);
      onSessionChange(refreshedSession);
      setAdjudicationNotice(
        response.model_effect.status === "pending_authority_revalidation"
          ? "更正已记录，但未改变掌握度或教学阶段：仍需已认证教师与 rubric 权威重新核验。"
          : response.model_effect.status === "applied_supersede_replay"
            ? response.authenticated_teacher && decision === "correct"
              ? "认证教师更正已由服务端重新核验 rubric 权威；原证据已 supersede，并精确重放目标知识点。"
              : "证据已安全作废，并从初始先验重放该知识点的其余有效证据。"
            : "复核决定已持久化；学生模型未发生变化。",
      );
    } catch (caught) {
      setAdjudicationError(caught instanceof Error ? caught.message : "复核决定失败");
    } finally {
      setAdjudicationBusy(null);
    }
  }, [adjudicationSignals, adjudicationTokens, authorityUi.actionsAllowed, onSessionChange, session]);

  React.useEffect(() => {
    if (!authorityUi.actionsAllowed) setAdjudicationTokens({});
  }, [authorityUi.actionsAllowed]);

  React.useEffect(() => {
    try {
      const stored = window.localStorage.getItem("teachlab.console.preferences");
      if (stored) {
        const parsed = JSON.parse(stored) as {theme?: string; density?: string; reduceMotion?: boolean};
        if (parsed.theme === "Light" || parsed.theme === "Dark") setTheme(parsed.theme);
        if (parsed.density === "Default" || parsed.density === "Compact") setDensity(parsed.density);
        if (typeof parsed.reduceMotion === "boolean") setReduceMotion(parsed.reduceMotion);
      }
    } catch {
      // A malformed local preference should never block the inspector.
    } finally {
      setPreferencesReady(true);
    }
  }, []);

  React.useEffect(() => {
    if (settingsRequest > 0) setTab("appearance");
  }, [settingsRequest]);

  React.useEffect(() => {
    if (resourcesRequest > 0) setTab("resources");
  }, [resourcesRequest]);

  React.useEffect(() => {
    if (!preferencesReady) return;
    window.localStorage.setItem("teachlab.console.preferences", JSON.stringify({theme, density, reduceMotion, webSearchEnabled}));
    document.documentElement.dataset.consoleTheme = theme.toLowerCase();
    document.documentElement.dataset.consoleDensity = density.toLowerCase();
    document.documentElement.dataset.consoleReduceMotion = String(reduceMotion);
  }, [density, preferencesReady, reduceMotion, theme, webSearchEnabled]);

  React.useEffect(() => {
    const media = window.matchMedia("(max-width: 1260px)");
    const sync = (event: MediaQueryList | MediaQueryListEvent) => setIsMobile(event.matches);
    sync(media);
    media.addEventListener("change", sync);
    return () => media.removeEventListener("change", sync);
  }, []);

  React.useEffect(() => {
    if (!open || !isMobile) return;
    const previous = document.activeElement as HTMLElement | null;
    closeButtonRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab" || !panelRef.current) return;
      const focusable = Array.from(panelRef.current.querySelectorAll<HTMLElement>(
        'button:not([disabled]), summary, [href], input:not([disabled]), textarea:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'
      )).filter((element) => element.getClientRects().length > 0);
      if (!focusable.length) return;
      event.preventDefault();
      const activeIndex = focusable.indexOf(document.activeElement as HTMLElement);
      const nextIndex = event.shiftKey
        ? activeIndex <= 0 ? focusable.length - 1 : activeIndex - 1
        : activeIndex < 0 || activeIndex === focusable.length - 1 ? 0 : activeIndex + 1;
      focusable[nextIndex]?.focus();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      previous?.focus?.();
    };
  }, [isMobile, onClose, open]);

  const mastery = session?.student_state?.knowledge_mastery ?? {};
  const setupProfile = session?.setup_snapshot?.student_profile;
  const preferences = session?.profile_summary?.preferences ?? setupProfile?.preferences ?? [];
  const initialMastery = session?.profile_summary?.initial_mastery ?? setupProfile?.initial_mastery ?? {};
  const misconceptions = setupProfile?.known_misconceptions ?? [];
  const accessibilityNeeds = setupProfile?.accessibility_needs ?? [];
  const action = session?.next_action ?? session?.current_action;
  const historicalConsentReceipts = consentReceipts.filter(
    (receipt) => !consentReceiptIsCurrent(receipt, remoteConsent?.policies),
  );
  const lessonProgress = visibleLessonProgress(session?.lesson_progress ?? action?.lesson_phase);
  const teachingResources = teachingResourcesProp ?? session?.teaching_resources ?? [];
  const runtimeLines = session ? [
    `$ teacher-agent session ${session.session_id.slice(0, 10)}`,
    `round ${session.rounds_completed ?? 0} · context v${session.context_version}`,
    `fallback: ${String(session.agent_runtime?.fallback_count ?? 0)} · status: ${session.status ?? "active"}`,
    "lifecycle: version guards active"
  ] : ["$ teacher-agent connecting", "waiting for bootstrap", "fallback: —", "lifecycle: pending"];

  return (
    <>
      <button
        type="button"
        aria-label="关闭检查器"
        onClick={onClose}
        className={cn(
          "fixed inset-0 z-40 bg-black/25 transition-opacity min-[1261px]:hidden",
          open && isMobile ? "pointer-events-auto opacity-100" : "pointer-events-none opacity-0"
        )}
      />
      <div
        className={cn(
          "relative h-full shrink-0 overflow-visible transition-[width] duration-200 min-[1261px]:w-[490px] max-[1260px]:fixed max-[1260px]:inset-y-0 max-[1260px]:right-0 max-[1260px]:z-50 max-[1260px]:w-0",
          !open && "min-[1261px]:w-0 max-[1260px]:pointer-events-none"
        )}
      >
      <aside
        ref={panelRef}
        role={isMobile ? "dialog" : "complementary"}
        aria-modal={isMobile ? true : undefined}
        aria-label="会话检查器"
        aria-hidden={!open}
        inert={!open}
        data-inspector-theme={theme.toLowerCase()}
        className={cn(
          "absolute inset-y-0 right-0 grid w-[490px] grid-cols-[68px_minmax(0,1fr)] grid-rows-[48px_minmax(0,1fr)] overflow-hidden border-l border-[var(--inspector-border)] bg-[var(--inspector-bg)] text-[var(--inspector-text)] shadow-[-18px_0_48px_var(--app-shadow)] transition-transform max-[1260px]:w-[min(520px,calc(100vw-16px))] max-[560px]:w-[calc(100vw-8px)] max-[1260px]:grid-cols-[58px_minmax(0,1fr)] max-[1260px]:shadow-[-24px_0_80px_var(--app-shadow)]",
          reduceMotion ? "duration-0" : "duration-200",
          open ? "translate-x-0" : "translate-x-full"
        )}
      >
        <header className="col-span-2 flex h-12 items-center border-b border-[var(--inspector-border)] bg-[var(--inspector-rail)] px-3 text-[var(--inspector-text)]">
          <button type="button" onClick={() => setTab("profile")} className="grid size-8 place-items-center rounded-lg text-[var(--inspector-muted)] hover:bg-[var(--inspector-control)] hover:text-[var(--inspector-text)]" aria-label="返回学生画像" title="返回学生画像"><ChevronLeft className="size-4" /></button>
          <span className="ml-2 min-w-0 truncate text-xs text-[var(--inspector-muted)]">TeachLab / session inspector</span>
          <Button ref={closeButtonRef} variant="subtle" size="icon" className="ml-auto" onClick={onClose} aria-label="收起右侧栏"><X className="size-4" /></Button>
        </header>

        <nav className="row-start-2 flex min-h-0 flex-col items-center gap-3 border-r border-[var(--inspector-border)] bg-[var(--inspector-rail)] py-4">
          <span role="img" className="mb-2 grid size-11 place-items-center rounded-xl bg-[var(--app-success)] text-[var(--app-bg)]" aria-label="TeachLab"><BrainCircuit className="size-5" /></span>
          {([
            ["profile", UserRound, "学生画像"],
            ["state", Home, "学情"],
            ["resources", BookOpen, "教学资源"],
            ["method", BrainCircuit, "方法"],
            ["review", Scale, "证据复核"],
            ["consent", ShieldCheck, "同意中心"],
            ["safeguarding", ShieldAlert, "安全保障"],
            ["runtime", Gauge, "运行"]
          ] as const).map(([value, Icon, label]) => (
            <button
              key={value}
              type="button"
              aria-label={label}
              onClick={() => setTab(value)}
              title={label}
              className={cn("grid size-11 place-items-center rounded-xl text-[var(--inspector-muted)] transition", tab === value && "bg-[var(--inspector-card)] text-[var(--app-success)] shadow-sm")}
            >
              <Icon className="size-5" />
            </button>
          ))}
          <button type="button" onClick={() => setTab("appearance")} className={cn("mt-auto grid size-11 place-items-center rounded-xl text-[var(--inspector-muted)] hover:bg-[var(--inspector-card)]", tab === "appearance" && "bg-[var(--inspector-card)] text-[var(--app-success)] shadow-sm")} aria-label="打开检查器设置" title="设置"><Settings className="size-5" /></button>
        </nav>

        <div className={cn("row-start-2 min-h-0 overflow-y-auto px-10 max-[560px]:px-4", density === "Compact" ? "py-4" : "py-8")}>
          {tab === "profile" && (
            <section>
              <h2 className="text-2xl font-semibold tracking-tight">学生画像</h2>
              <p className="mt-2 text-sm text-[var(--inspector-muted)]">来自当前后端会话的只读画像与初始设定。</p>
              <Separator className="my-7 bg-[var(--inspector-border)]" />
              {!session ? (
                <div className="rounded-xl border border-dashed border-[var(--inspector-border)] p-5 text-sm leading-6 text-[var(--inspector-muted)]">发送第一条学习目标后，这里会显示真实学生画像。</div>
              ) : (
                <div className="grid gap-5">
                  <div className="rounded-xl border border-[var(--inspector-border)] bg-[var(--inspector-card)] p-5 shadow-sm">
                    <div className="flex items-center gap-3">
                      <span className="grid size-10 place-items-center rounded-full bg-[var(--app-success-soft)] text-[var(--app-success)]"><UserRound className="size-5" /></span>
                      <div className="min-w-0">
                        <strong className="block truncate">{session.profile_summary?.display_name ?? "本地学生"}</strong>
                        <span className="text-sm text-[var(--inspector-muted)]">{session.profile_summary?.learner_level ?? setupProfile?.learner_level ?? "未标注学习阶段"}</span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-sm font-semibold">学习偏好</h3>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {preferences.length ? preferences.map((preference) => <span key={preference} className="rounded-full border border-[var(--inspector-border)] bg-[var(--inspector-card)] px-3 py-1.5 text-xs">{preference}</span>) : <span className="text-sm text-[var(--inspector-muted)]">暂无偏好记录</span>}
                    </div>
                  </div>

                  <div>
                    <h3 className="text-sm font-semibold">初始掌握度</h3>
                    <div className="mt-3 grid gap-2">
                      {Object.entries(initialMastery).length ? Object.entries(initialMastery).map(([key, value]) => (
                        <div key={key} className="flex items-center justify-between rounded-xl border border-[var(--inspector-border)] bg-[var(--inspector-card)] px-4 py-3 text-sm">
                          <span>{masteryLabels[key] ?? key}</span><strong>{Math.round(value * 100)}%</strong>
                        </div>
                      )) : <span className="text-sm text-[var(--inspector-muted)]">暂无初始掌握度</span>}
                    </div>
                  </div>

                  <div>
                    <h3 className="text-sm font-semibold">已知易错点</h3>
                    <div className="mt-3 grid gap-2">
                      {misconceptions.length ? misconceptions.map((item, index) => (
                        <div key={`${item.tag ?? "misconception"}-${index}`} className="rounded-xl border border-[var(--inspector-border)] bg-[var(--inspector-card)] p-4 text-sm">
                          <strong>{item.tag ?? "待核验误区"}</strong>
                          {item.description && <p className="mt-1 leading-6 text-[var(--inspector-muted)]">{item.description}</p>}
                        </div>
                      )) : <span className="text-sm text-[var(--inspector-muted)]">暂无已知易错点</span>}
                    </div>
                  </div>

                  {accessibilityNeeds.length > 0 && (
                    <div>
                      <h3 className="text-sm font-semibold">交互需要</h3>
                      <p className="mt-2 text-sm leading-6 text-[var(--inspector-muted)]">{accessibilityNeeds.join(" · ")}</p>
                    </div>
                  )}
                </div>
              )}
            </section>
          )}

          {tab === "appearance" && (
            <section>
              <h2 className="text-2xl font-semibold tracking-tight">外观</h2>
              <Separator className="mt-7 bg-[var(--inspector-border)]" />
              <SettingRow compact={density === "Compact"} title="主题" description="选择教学工作台的显示方式。">
                <SegmentedControl label="主题" value={theme} options={["Light", "Dark"]} optionLabels={{Light: "浅色", Dark: "深色"}} onChange={setTheme} />
              </SettingRow>
              <Separator className="bg-[var(--inspector-border)]" />
              <SettingRow compact={density === "Compact"} title="密度" description="为高频使用者压缩行高与间距。">
                <SegmentedControl label="密度" value={density} options={["Default", "Compact"]} optionLabels={{Default: "默认", Compact: "紧凑"}} onChange={setDensity} />
              </SettingRow>
              <Separator className="bg-[var(--inspector-border)]" />
              <SettingRow compact={density === "Compact"} title="减少动画" description="关闭 drawer、流式光标与状态过渡。">
                <Switch checked={reduceMotion} onCheckedChange={setReduceMotion} label="减少动画" />
              </SettingRow>
              <Separator className="bg-[var(--inspector-border)]" />
              <SettingRow compact={density === "Compact"} title="Chat 联网搜索" description="允许 DeepSeek 在问题需要最新信息时按需搜索，并显示来源。">
                <Switch checked={webSearchEnabled} onCheckedChange={onWebSearchEnabledChange} label="Chat 联网搜索" />
              </SettingRow>
              {visualSemantics?.available && (
                <>
                  <Separator className="bg-[var(--inspector-border)]" />
                  <SettingRow
                    compact={density === "Compact"}
                    title="图片视觉语义"
                    description={visualSemantics.sends_raw_media_remotely
                      ? "开启后，图片会在每次上传前核验专用服务器收据，再发送给远程视觉服务商。"
                      : "开启后只在本机分析图片布局；不把原图发送到远程服务。"}
                  >
                    <Switch checked={visualAnalysisEnabled} onCheckedChange={onVisualAnalysisEnabledChange} label="图片视觉语义" />
                  </SettingRow>
                </>
              )}
            </section>
          )}

          {tab === "state" && (
            <section>
              <h2 className="text-2xl font-semibold tracking-tight">当前学情</h2>
              <Separator className="my-7 bg-[var(--inspector-border)]" />
              <div className="grid gap-4">
                {lessonProgress && (
                  <div
                    className="rounded-xl border border-[var(--inspector-border)] bg-[var(--inspector-card)] p-4 shadow-sm"
                    aria-label={`当前教学阶段：${lessonProgress.label}，第 ${lessonProgress.index} 阶段，共 6 阶段`}
                  >
                    <div className="flex items-center justify-between gap-4">
                      <div className="min-w-0">
                        <p className="text-xs text-[var(--inspector-muted)]">当前教学阶段</p>
                        <strong className="mt-1 flex items-center gap-2 text-base">
                          <span aria-hidden="true" className="size-1.5 shrink-0 rounded-full bg-[var(--app-accent)]" />
                          <span className="truncate">{lessonProgress.label}</span>
                        </strong>
                      </div>
                      <span className="shrink-0 text-sm tabular-nums text-[var(--inspector-muted)]">{lessonProgress.index} / 6</span>
                    </div>
                    {lessonProgress.currentConcept && <p className="mt-3 line-clamp-2 text-xs leading-5 text-[var(--inspector-muted)]">当前知识点 · {lessonProgress.currentConcept}</p>}
                  </div>
                )}
                {[["前置知识", mastery.prerequisite], ["概念理解", mastery.conceptual], ["操作过程", mastery.procedural], ["迁移能力", mastery.transfer]].map(([label, value]) => (
                  <div key={String(label)} className="flex items-center justify-between rounded-xl border border-[var(--inspector-border)] bg-[var(--inspector-card)] p-4 text-sm shadow-sm">
                    <span>{label}</span><strong>{typeof value === "number" ? `${Math.round(value * 100)}%` : "—"}</strong>
                  </div>
                ))}
                {learningError && <p className="rounded-lg border border-red-400/40 bg-red-500/10 p-3 text-xs leading-5 text-red-300" role="alert">{learningError}</p>}
                {learningNotice && <p className="rounded-lg border border-[var(--app-success)]/40 bg-[var(--app-success-soft)] p-3 text-xs leading-5 text-[var(--app-success)]" role="status">{learningNotice}</p>}

                <details open className="rounded-xl border border-[var(--inspector-border)] bg-[var(--inspector-card)] p-4 shadow-sm">
                  <summary className="cursor-pointer text-sm font-semibold">答题前自评（JOL） · {learnerJol}%</summary>
                  <p className="mt-2 text-xs leading-5 text-[var(--inspector-muted)]">只记录答题前的把握和准备策略；不记录答案，也不改变掌握度。</p>
                  <label className="mt-3 grid gap-2 text-xs text-[var(--inspector-muted)]">
                    我觉得自己能正确回答的概率
                    <input type="range" min={0} max={100} step={1} value={learnerJol} onChange={(event) => setLearnerJol(Number(event.target.value))} aria-valuetext={`${learnerJol}%`} className="w-full accent-[var(--app-success)]" />
                  </label>
                  <fieldset className="mt-3">
                    <legend className="text-xs text-[var(--inspector-muted)]">准备使用的策略（1–3 个）</legend>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {metacognitionStrategies.map((strategy) => {
                        const selected = metacognitionStrategiesSelected.includes(strategy);
                        return (
                          <button key={strategy} type="button" aria-pressed={selected} disabled={!selected && metacognitionStrategiesSelected.length >= 3} onClick={() => setMetacognitionStrategiesSelected((current) => selected ? current.filter((item) => item !== strategy) : [...current, strategy])} className={cn("rounded-full border px-3 py-1.5 text-xs disabled:opacity-40", selected ? "border-[var(--app-success)] bg-[var(--app-success-soft)] text-[var(--app-success)]" : "border-[var(--inspector-border)] text-[var(--inspector-muted)]")}>
                            {metacognitionStrategyLabels[strategy]}
                          </button>
                        );
                      })}
                    </div>
                  </fieldset>
                  {activeLearningReviews.length > 0 && <p className="mt-3 text-xs text-[var(--app-warning)]">当前复习会话会由服务端自动绑定其租约；浏览器不提交结果或知识点。</p>}
                  <Button className="mt-3 w-full" size="sm" disabled={!session || learningBusy !== null || metacognitionStrategiesSelected.length < 1} onClick={() => void submitMetacognitionPrediction()}>{learningBusy === "prediction" ? "记录中…" : "在答题前记录预测"}</Button>
                </details>

                <details className="rounded-xl border border-[var(--inspector-border)] bg-[var(--inspector-card)] p-4 shadow-sm">
                  <summary className="cursor-pointer text-sm font-semibold">预测与权威结果（{metacognitionItems.length}）</summary>
                  <p className="mt-2 text-xs leading-5 text-[var(--inspector-muted)]">刷新后仍可恢复；仅用服务端已提交的教学回合收据配对。</p>
                  <Button className="mt-3 w-full" variant="subtle" size="sm" disabled={!session || learningBusy !== null} onClick={() => void refreshLongTermLearning(true)}>刷新并自动配对</Button>
                  <div className="mt-3 grid gap-2">
                    {!metacognitionItems.length ? <p className="text-xs text-[var(--inspector-muted)]">当前会话还没有预测。</p> : metacognitionItems.map((item) => (
                      <article key={item.prediction_event_id} className="rounded-lg border border-[var(--inspector-border)] bg-[var(--inspector-control)] p-3 text-xs">
                        <div className="flex justify-between gap-2"><strong>{item.knowledge_component_id} · {item.learner_jol_percent}%</strong><span className={item.status === "paired" ? "text-[var(--app-success)]" : "text-[var(--app-warning)]"}>{item.status === "paired" ? "已配对" : "待收据"}</span></div>
                        <p className="mt-1 text-[var(--inspector-muted)]">{item.strategy_codes.map((strategy) => metacognitionStrategyLabels[strategy]).join(" · ")}</p>
                        {item.feedback ? <div className="mt-2 rounded-lg bg-[var(--app-success-soft)] p-2 leading-5"><strong>{item.feedback.calibration_classification === "overconfident" ? "自评偏高" : item.feedback.calibration_classification === "underconfident" ? "自评偏低" : "自评较准"} · 权威 {item.feedback.actual_score_percent ?? "—"}%</strong><p className="text-[var(--inspector-muted)]">{item.feedback.message_zh}</p></div> : <Button className="mt-2 w-full" variant="subtle" size="sm" disabled={learningBusy !== null} onClick={() => void explicitlyPairPrediction(item.prediction_event_id)}>{learningBusy === item.prediction_event_id ? "核验中…" : "核验服务端收据"}</Button>}
                      </article>
                    ))}
                  </div>
                </details>

                <details className="rounded-xl border border-[var(--inspector-border)] bg-[var(--inspector-card)] p-4 shadow-sm">
                  <summary className="cursor-pointer text-sm font-semibold">到期复习（{dueLearningReviews.length}）</summary>
                  <p className="mt-2 text-xs leading-5 text-[var(--inspector-muted)]">到期时间由服务端推导；没有客户端结果、掌握度或评分输入。</p>
                  {activeLearningReviews.map((review) => (
                    <article key={review.active_lease_id ?? review.knowledge_component_id} className="mt-3 rounded-lg border border-[var(--app-warning)] bg-[var(--app-warning-soft)] p-3 text-xs">
                      <strong>{review.knowledge_component_id} · 复习租约活动</strong>
                      <p className="mt-1 text-[var(--inspector-muted)]">到期 {review.lease_expires_at_utc ? new Date(review.lease_expires_at_utc).toLocaleString() : "—"}</p>
                      {review.review_session_id && review.review_session_id !== session?.session_id && (
                        <Button className="mt-2 w-full" variant="subtle" size="sm" disabled={learningBusy !== null} onClick={() => void continueLearningReview(review)}>继续复习</Button>
                      )}
                      {(!review.review_session_id || review.review_session_id === session?.session_id) && (
                        <Button className="mt-2 w-full" variant="subtle" size="sm" disabled={learningBusy !== null} onClick={() => void releaseLearningReviewFromView(review)}>释放租约</Button>
                      )}
                    </article>
                  ))}
                  <div className="mt-3 grid gap-2">
                    {dueLearningReviews.length ? dueLearningReviews.map((review) => (
                      <article key={review.review_id} className="rounded-lg border border-[var(--inspector-border)] bg-[var(--inspector-control)] p-3 text-xs">
                        <strong>{review.knowledge_component_id}</strong>
                        <p className="mt-1 text-[var(--inspector-muted)]">到期 {new Date(review.due_at_utc).toLocaleString()} · {review.claim_state === "expired" ? "旧租约已过期" : "待领取"}</p>
                        <Button className="mt-2 w-full" variant="subtle" size="sm" disabled={learningBusy !== null || activeLearningReviews.length > 0} onClick={() => void claimLearningReview(review)}>{learningBusy === review.review_id ? "正在打开…" : "开始复习"}</Button>
                      </article>
                    )) : activeLearningReviews.length === 0 && <p className="text-xs text-[var(--inspector-muted)]">当前没有到期复习。</p>}
                  </div>
                </details>
              </div>
            </section>
          )}

          {tab === "resources" && (
            <section>
              <h2 className="text-2xl font-semibold tracking-tight">教学资源</h2>
              <p className="mt-2 text-sm leading-6 text-[var(--inspector-muted)]">通过输入框左下角“+”导入。原文件只在本机短暂处理，DeepSeek 只接收受限文本。</p>
              <Separator className="my-7 bg-[var(--inspector-border)]" />
              {!teachingResources.length ? (
                <div className="rounded-xl border border-dashed border-[var(--inspector-border)] p-5 text-sm leading-6 text-[var(--inspector-muted)]">尚未导入 PPT、PDF、教学文稿或图片资源。</div>
              ) : (
                <div className="grid gap-3">
                  {teachingResources.map((resource) => (
                    <TeachingResourceCard key={resource.resource_id} resource={resource} teacherAuthority={teacherAuthority} onReviewed={onResourceReviewed} />
                  ))}
                </div>
              )}
            </section>
          )}

          {tab === "method" && (
            <section>
              <h2 className="text-2xl font-semibold tracking-tight">教学方法</h2>
              <Separator className="my-7 bg-[var(--inspector-border)]" />
              <div className="rounded-xl border border-[var(--inspector-border)] bg-[var(--inspector-card)] p-5 shadow-sm">
                <div className="flex items-center gap-3"><Bot className="size-5 text-[var(--app-success)]" /><strong>{String(action?.primary_skill?.name ?? "等待 Skill 选择")}</strong></div>
                <p className="mt-3 text-sm leading-6 text-[var(--inspector-muted)]">允许核验明确的自解释证据；不读取 gold，不自动提高掌握度。</p>
              </div>
            </section>
          )}

          {tab === "review" && (
            <section>
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h2 className="text-2xl font-semibold tracking-tight">证据复核</h2>
                  <p className="mt-2 text-sm leading-6 text-[var(--inspector-muted)]">队列只公开理由、定位符、知识点与内容哈希，不公开学生原文。</p>
                </div>
                <Button variant="subtle" size="sm" onClick={() => void refreshAdjudications()} disabled={!session || adjudicationBusy !== null}>刷新</Button>
              </div>
              <div className="mt-4 rounded-xl border border-[var(--app-warning)] bg-[var(--app-warning-soft)] p-4 text-xs leading-5 text-[var(--app-warning)]" role="note">
                <strong>{authorityUi.title}</strong>
                {authorityUi.authenticatedTeacher
                  ? "：服务端已从认证会话核验教师角色。Correct 仍须绑定当前 payload、证据、知识点与 rubric 的短期签名收据，再 supersede 并精确重放；该收据不代表个人不可否认签名。"
                  : authorityUi.mode === "local_python"
                    ? "：当前身份为 local_operator_not_authenticated。Approve 仅确认、不改模型；Correct 只进入权威复核等待态；Abstain 才会作废该证据并精确重放目标知识点。"
                    : authorityUi.mode === "authenticated_apps_api"
                      ? "：当前认证会话没有配置允许的教师角色，领取与裁决失败关闭。浏览器不能提交角色、actor 或 authority receipt。"
                      : "：bootstrap 权威状态缺失或不一致，领取与裁决失败关闭。"}
              </div>
              {adjudicationError && <p className="mt-3 rounded-lg border border-red-400/40 bg-red-500/10 p-3 text-xs leading-5 text-red-300" role="alert">{adjudicationError}</p>}
              {adjudicationNotice && <p className="mt-3 rounded-lg border border-[var(--app-success)]/40 bg-[var(--app-success-soft)] p-3 text-xs leading-5 text-[var(--app-success)]" role="status">{adjudicationNotice}</p>}
              <Separator className="my-6 bg-[var(--inspector-border)]" />
              {session && adjudicationCandidates.length > 0 && (
                <div className="mb-6">
                  <h3 className="text-sm font-semibold">可加入复核的权威证据</h3>
                  <p className="mt-1 text-xs leading-5 text-[var(--inspector-muted)]">这些候选项由服务器从 KC 账本生成；加入时会再次验证不可变历史。</p>
                  <div className="mt-3 grid gap-2">
                    {adjudicationCandidates.map((candidate) => {
                      const queued = adjudicationItems.some((item) => item.source.round_number === candidate.history_round && item.target_kc_ids.includes(candidate.knowledge_component_id));
                      return (
                        <div key={candidate.evidence_locator_sha256} className="flex items-center justify-between gap-3 rounded-lg border border-[var(--inspector-border)] bg-[var(--inspector-card)] p-3 text-xs">
                          <div className="min-w-0">
                            <strong className="block truncate">{candidate.knowledge_component_label ?? candidate.knowledge_component_id}</strong>
                            <span className="text-[var(--inspector-muted)]">round {candidate.history_round} · {candidate.question_id ?? "question locator"}</span>
                          </div>
                          <Button variant="subtle" size="sm" disabled={queued || adjudicationBusy === candidate.evidence_locator_sha256} onClick={() => void enqueueReview(candidate)}>{queued ? "已入队" : "加入"}</Button>
                        </div>
                      );
                    })}
                  </div>
                  <Separator className="mt-6 bg-[var(--inspector-border)]" />
                </div>
              )}
              {!session ? (
                <p className="text-sm text-[var(--inspector-muted)]">先开始一个教学会话，再查看其复核队列。</p>
              ) : !adjudicationItems.length ? (
                <div className="rounded-xl border border-dashed border-[var(--inspector-border)] p-5 text-sm leading-6 text-[var(--inspector-muted)]">当前会话没有待复核证据。</div>
              ) : (
                <div className="grid gap-4">
                  {adjudicationItems.map((item) => {
                    const token = adjudicationTokens[item.item_id];
                    const busy = adjudicationBusy === item.item_id;
                    const terminal = item.status === "decided" || item.status === "cancelled";
                    return (
                      <article key={item.item_id} className="rounded-xl border border-[var(--inspector-border)] bg-[var(--inspector-card)] p-4 shadow-sm">
                        <div className="flex items-center justify-between gap-3">
                          <strong className="truncate text-sm" title={item.item_id}>{item.target_kc_ids.join(" · ")}</strong>
                          <span className="shrink-0 rounded-full border border-[var(--inspector-border)] px-2 py-1 text-[10px] uppercase">{item.status} · v{item.version}</span>
                        </div>
                        <dl className="mt-3 grid gap-2 text-xs leading-5 text-[var(--inspector-muted)]">
                          <div><dt className="inline font-medium text-[var(--inspector-text)]">理由：</dt><dd className="inline">{item.review_reason}</dd></div>
                          <div><dt className="inline font-medium text-[var(--inspector-text)]">定位：</dt><dd className="inline">round {item.source.round_number} · {item.source.question_id}</dd></div>
                          <div><dt className="inline font-medium text-[var(--inspector-text)]">证据：</dt><dd className="inline font-mono">{item.original.evidence_sha256.slice(0, 16)}…</dd></div>
                          <div><dt className="inline font-medium text-[var(--inspector-text)]">Rubric：</dt><dd className="inline break-all">{item.original.rubric_id}</dd></div>
                        </dl>
                        {!terminal && !token && item.status === "pending" && (
                          <Button className="mt-4 w-full" variant="subtle" size="sm" disabled={busy || !authorityUi.actionsAllowed} onClick={() => void claimReview(item)}>领取复核</Button>
                        )}
                        {!terminal && item.status === "claimed" && !token && (
                          <p className="mt-4 text-xs leading-5 text-[var(--inspector-muted)]">此项已被领取；claim token 不会持久化到浏览器。租约到期后可重新领取。</p>
                        )}
                        {!terminal && token && (
                          <div className="mt-4 grid gap-3">
                            <label className="grid gap-1 text-xs text-[var(--inspector-muted)]">
                              {authorityUi.authenticatedTeacher ? "认证更正标签（服务端将重新核验 rubric）" : "Correct 的建议标签（仍不会直接改掌握度）"}
                              <select
                                className="rounded-lg border border-[var(--inspector-border)] bg-[var(--inspector-control)] px-3 py-2 text-[var(--inspector-text)]"
                                value={adjudicationSignals[item.item_id] ?? "partial"}
                                onChange={(event) => setAdjudicationSignals((current) => ({...current, [item.item_id]: event.target.value as "correct" | "partial" | "misconception"}))}
                              >
                                <option value="correct">correct</option>
                                <option value="partial">partial</option>
                                <option value="misconception">misconception</option>
                              </select>
                            </label>
                            <div className="grid grid-cols-3 gap-2">
                              <Button variant="subtle" size="sm" disabled={busy || !authorityUi.actionsAllowed} onClick={() => void decideReview(item, "approve")}>Approve</Button>
                              <Button variant="subtle" size="sm" disabled={busy || !authorityUi.actionsAllowed} onClick={() => void decideReview(item, "correct")}>{authorityUi.authenticatedTeacher ? "认证更正" : "Correct"}</Button>
                              <Button variant="subtle" size="sm" disabled={busy || !authorityUi.actionsAllowed} onClick={() => void decideReview(item, "abstain")}>Abstain</Button>
                            </div>
                          </div>
                        )}
                        {item.decision?.kind && <p className="mt-4 text-xs text-[var(--inspector-muted)]">决定：{item.decision.kind} · {item.decision.reason_code}</p>}
                        {item.cancellation && <p className="mt-4 text-xs text-[var(--inspector-muted)]">已因证据删除取消。</p>}
                      </article>
                    );
                  })}
                </div>
              )}
            </section>
          )}

          {tab === "consent" && (
            <section>
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h2 className="text-2xl font-semibold tracking-tight">同意中心</h2>
                  <p className="mt-2 text-sm leading-6 text-[var(--inspector-muted)]">收据由本机后端签发；浏览器布尔值和旧 localStorage 记录都不是授权。</p>
                </div>
                <Button variant="subtle" size="sm" onClick={() => void refreshConsents()} disabled={consentBusy !== null}>刷新</Button>
              </div>
              {consentError && <p className="mt-4 rounded-lg border border-red-400/40 bg-red-500/10 p-3 text-xs leading-5 text-red-300" role="alert">{consentError}</p>}
              {consentNotice && <p className="mt-4 rounded-lg border border-[var(--app-success)]/40 bg-[var(--app-success-soft)] p-3 text-xs leading-5 text-[var(--app-success)]" role="status">{consentNotice}</p>}
              <Separator className="my-6 bg-[var(--inspector-border)]" />

              <div className="rounded-xl border border-[var(--inspector-border)] bg-[var(--inspector-card)] p-4">
                <strong className="text-sm">年龄与监护策略由组织服务器管理</strong>
                <p className="mt-1 text-xs leading-5 text-[var(--inspector-muted)]">浏览器不能自行声明成年，也不能自行签署“已核验监护人/学校政策”。未知或不符合组织政策时，远程处理会被服务器拒绝。</p>
              </div>

              <div className="mt-6 grid gap-3">
                {(remoteConsent?.policies ?? []).map((policy) => {
                  const active = consentReceipts.find((receipt) => receipt.purpose === policy.purpose
                    && consentReceiptIsCurrent(receipt, [policy]));
                  return (
                    <article key={policy.purpose} className="rounded-xl border border-[var(--inspector-border)] bg-[var(--inspector-card)] p-4 shadow-sm">
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <strong className="block text-sm">{consentPurposeLabels[policy.purpose]}</strong>
                          <p className="mt-1 break-words text-xs leading-5 text-[var(--inspector-muted)]">{policy.provider_id} · {policy.processing_region} · 保留 {policy.provider_retention_days} 天</p>
                          <p className="mt-1 text-[10px] leading-4 text-[var(--inspector-muted)]">{policy.data_categories.join(" · ")}</p>
                        </div>
                        <span className={cn("shrink-0 rounded-full border px-2 py-1 text-[10px]", active ? "border-[var(--app-success)] text-[var(--app-success)]" : "border-[var(--inspector-border)] text-[var(--inspector-muted)]")}>{active ? "ACTIVE" : "未授权"}</span>
                      </div>
                      {active ? (
                        <div className="mt-3 flex items-end justify-between gap-3 text-[10px] leading-4 text-[var(--inspector-muted)]">
                          <span>到期 {new Date(active.expires_at_utc).toLocaleString()}<br />未成年策略：{active.guardian_or_school_policy}</span>
                          <Button variant="subtle" size="sm" disabled={consentBusy !== null} onClick={() => void revokeConsent(active)}>撤销</Button>
                        </div>
                      ) : (
                        <Button className="mt-3 w-full" variant="subtle" size="sm" disabled={consentBusy !== null || !remoteConsent?.configured || !policy.remote_processing_eligible} onClick={() => void grantConsent(policy.purpose)}>{consentBusy === policy.purpose ? "签发中…" : policy.remote_processing_eligible ? "签发 30 天收据" : "组织策略未授权"}</Button>
                      )}
                    </article>
                  );
                })}
                {remoteConsent?.configured && !(remoteConsent.policies ?? []).length && (
                  <p className="rounded-xl border border-dashed border-[var(--inspector-border)] p-4 text-sm leading-6 text-[var(--inspector-muted)]">当前没有需要远程授权的服务；本地视觉分析不会发送原图。</p>
                )}
              </div>

              {historicalConsentReceipts.length > 0 && (
                <details className="mt-6 text-xs text-[var(--inspector-muted)]">
                  <summary className="cursor-pointer">已撤销、过期或策略失效的收据（审计保留）</summary>
                  <div className="mt-3 grid gap-2">
                    {historicalConsentReceipts.map((receipt) => (
                      <div key={receipt.consent_id} className="rounded-lg border border-[var(--inspector-border)] p-3">
                        {consentPurposeLabels[receipt.purpose]} · {receipt.provider_id}<br />
                        状态：{receipt.status === "revoked" ? "已撤销" : Date.parse(receipt.expires_at_utc) <= Date.now() ? "已过期" : "服务商策略已变化"} · 未成年策略：{receipt.guardian_or_school_policy}<br />
                        区域：{receipt.processing_region} · 保留 {receipt.provider_retention_days} 天 · {receipt.data_categories.join(" · ")}
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </section>
          )}

          {tab === "safeguarding" && (
            <section>
              <h2 className="text-2xl font-semibold tracking-tight">安全保障案例</h2>
              <p className="mt-2 text-sm leading-6 text-[var(--inspector-muted)]">仅供当前经组织目录实时授权的 safeguarding 人员。浏览器角色、学生原文和人工输入的学习者标识均不被接受。</p>
              <Separator className="my-7 bg-[var(--inspector-border)]" />
              <label className="grid gap-2 text-sm font-medium">
                中央系统路由凭据
                <input
                  type="password"
                  autoComplete="off"
                  spellCheck={false}
                  value={safeguardingRouteLocator}
                  disabled={safeguardingBusy !== null}
                  onChange={(event) => {
                    setSafeguardingRouteLocator(event.target.value.trim());
                    setSafeguardingCases([]);
                    setSafeguardingError(null);
                    setSafeguardingNotice(null);
                  }}
                  placeholder="sgr1_k…"
                  className="rounded-lg border border-[var(--inspector-border)] bg-[var(--inspector-control)] px-3 py-2 font-mono text-xs text-[var(--inspector-text)] outline-none focus:border-[var(--app-accent)] disabled:opacity-60"
                  aria-describedby="safeguarding-route-boundary"
                />
              </label>
              <p id="safeguarding-route-boundary" className="mt-2 text-[10px] leading-4 text-[var(--inspector-muted)]">凭据只保存在此面板内存，关闭检查器即清除；服务端会解密并强制校验同一租户。</p>
              <Button
                className="mt-4 w-full"
                size="sm"
                disabled={safeguardingBusy !== null || !validSafeguardingRouteLocator(safeguardingRouteLocator)}
                onClick={() => void refreshSafeguardingCases()}
              >{safeguardingBusy === "list" ? "正在读取…" : "读取内容无关案例"}</Button>

              {safeguardingError && <p className="mt-4 rounded-lg border border-red-400/40 bg-red-500/10 p-3 text-xs leading-5 text-red-300" role="alert">{safeguardingError}</p>}
              {safeguardingNotice && <p className="mt-4 rounded-lg border border-[var(--app-success)]/40 bg-[var(--app-success-soft)] p-3 text-xs leading-5 text-[var(--app-success)]" role="status">{safeguardingNotice}</p>}

              <div className="mt-5 grid gap-4">
                {safeguardingCases.map((item) => {
                  const deliveryNeedsAck = item.delivery_status === "pending" || item.delivery_status === "overdue";
                  const canClose = item.status === "acknowledged"
                    && (item.delivery_status === "acknowledged" || item.delivery_status === "escalation_unavailable");
                  return (
                    <article key={item.case_id} className="rounded-xl border border-[var(--inspector-border)] bg-[var(--inspector-card)] p-4 shadow-sm">
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <strong className="block font-mono text-xs">{item.case_id}</strong>
                          <p className="mt-1 text-xs text-[var(--inspector-muted)]">{item.category} · {item.severity} · v{item.version}</p>
                        </div>
                        <span className={cn(
                          "shrink-0 rounded-full border px-2 py-1 text-[10px]",
                          item.status === "closed"
                            ? "border-[var(--inspector-border)] text-[var(--inspector-muted)]"
                            : "border-[var(--app-warning)] bg-[var(--app-warning-soft)] text-[var(--app-warning)]",
                        )}>{item.status}</span>
                      </div>
                      <dl className="mt-3 grid gap-1 rounded-lg bg-[var(--inspector-control)] p-3 text-[10px] leading-4 text-[var(--inspector-muted)]">
                        <div><dt className="inline font-medium text-[var(--inspector-text)]">发现：</dt><dd className="inline"> {new Date(item.observed_at_utc).toLocaleString()}</dd></div>
                        <div><dt className="inline font-medium text-[var(--inspector-text)]">投递：</dt><dd className="inline"> {item.delivery_status}{item.sla_due_at_utc ? ` · SLA ${new Date(item.sla_due_at_utc).toLocaleString()}` : ""}</dd></div>
                        <div className="break-all"><dt className="inline font-medium text-[var(--inspector-text)]">内容 SHA-256：</dt><dd className="inline font-mono"> {item.content_sha256}</dd></div>
                        <div className="break-all"><dt className="inline font-medium text-[var(--inspector-text)]">范围 SHA-256：</dt><dd className="inline font-mono"> {item.scope_sha256}</dd></div>
                      </dl>
                      <div className="mt-3 grid grid-cols-1 gap-2 min-[561px]:grid-cols-2">
                        {deliveryNeedsAck && <Button variant="subtle" size="sm" disabled={safeguardingBusy !== null} onClick={() => void mutateSafeguardingCase(item, "delivery_acknowledge")}>确认中央队列接收</Button>}
                        {item.status === "open" && <Button variant="subtle" size="sm" disabled={safeguardingBusy !== null} onClick={() => void mutateSafeguardingCase(item, "case_acknowledge")}>确认人工接手</Button>}
                        {canClose && <Button size="sm" disabled={safeguardingBusy !== null} onClick={() => void mutateSafeguardingCase(item, "case_close")}>关闭案例</Button>}
                      </div>
                    </article>
                  );
                })}
                {validSafeguardingRouteLocator(safeguardingRouteLocator) && safeguardingBusy === null && safeguardingCases.length === 0 && !safeguardingError && (
                  <p className="rounded-xl border border-dashed border-[var(--inspector-border)] p-4 text-sm leading-6 text-[var(--inspector-muted)]">尚未读取案例，或当前路由范围没有案例。</p>
                )}
              </div>
            </section>
          )}

          {tab === "runtime" && (
            <section>
              <h2 className="text-2xl font-semibold tracking-tight">运行状态</h2>
              <Separator className="my-7 bg-[var(--inspector-border)]" />
              <div className="grid gap-3 text-sm">
                <div className="flex justify-between rounded-xl border border-[var(--inspector-border)] bg-[var(--inspector-card)] p-4"><span>模型</span><strong>{providerStatus?.model ?? providerStatus?.provider ?? "未连接"}</strong></div>
                <div className="flex justify-between rounded-xl border border-[var(--inspector-border)] bg-[var(--inspector-card)] p-4"><span>控制</span><strong>{session?.pending_skill_id ? "MANUAL" : "AUTO"}</strong></div>
                <div className="flex justify-between rounded-xl border border-[var(--inspector-border)] bg-[var(--inspector-card)] p-4"><span>Fallback</span><strong>{String(session?.agent_runtime?.fallback_count ?? "—")}</strong></div>
              </div>
              <div className="mt-5">
                <p className="mb-2 text-xs font-medium text-[var(--inspector-muted)]">会话状态快照</p>
                <pre aria-label="会话状态快照" className="max-h-56 overflow-auto whitespace-pre-wrap rounded-lg border border-[var(--app-border)] bg-[var(--app-surface-soft)] p-3 font-mono text-xs leading-6 text-[var(--app-text-soft)]">{runtimeLines.join("\n")}</pre>
              </div>
            </section>
          )}
        </div>
      </aside>
      </div>
    </>
  );
}
