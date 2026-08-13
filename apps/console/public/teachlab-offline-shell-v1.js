(() => {
  "use strict";

  const DATABASE = "teachlab-console-runtime";
  const PROJECT_STORE = "workspaces";
  const ACTIVE_PROJECT_KEY = "teachlab.learning-project.active-id";
  const ACCOUNT_SCOPE_KEY = "teachlab.account-cache-scope.v1";
  const ACCOUNT_SCOPE_PATTERN = /^acs1_[A-Za-z0-9_-]{43}$/;
  const ACCOUNT_TRANSITION_COOKIE_NAMES = new Set([
    "teachlab_account_transition",
    "__Host-teachlab_account_transition",
  ]);
  const MAX_VISIBLE_MESSAGES = 30;

  const byId = (id) => document.getElementById(id);
  const showFailure = (message) => {
    byId("offline-status").textContent = message;
    byId("offline-project").hidden = true;
  };
  const constantTimeEqual = (left, right) => {
    const maximum = Math.max(left.length, right.length);
    let mismatch = left.length ^ right.length;
    for (let index = 0; index < maximum; index += 1) {
      mismatch |= (left.charCodeAt(index) || 0) ^ (right.charCodeAt(index) || 0);
    }
    return mismatch === 0;
  };
  const accountTransitionPending = () => document.cookie.split(";")
    .map((part) => part.trim().split("=", 1)[0] || "")
    .some((name) => ACCOUNT_TRANSITION_COOKIE_NAMES.has(name));
  const openDatabase = () => new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  const readSnapshot = async (projectId) => {
    const database = await openDatabase();
    try {
      if (!database.objectStoreNames.contains(PROJECT_STORE)) return null;
      return await new Promise((resolve, reject) => {
        const transaction = database.transaction(PROJECT_STORE, "readonly");
        const request = transaction.objectStore(PROJECT_STORE).get(projectId);
        request.onsuccess = () => resolve(request.result ?? null);
        request.onerror = () => reject(request.error);
      });
    } finally {
      database.close();
    }
  };
  const validSnapshot = (value, projectId, accountScope) => {
    if (!value || value.schema !== "teachlab.offline_workspace_snapshot.v2") return false;
    if (!constantTimeEqual(String(value.accountCacheScope || ""), accountScope)) return false;
    if (value.projectId !== projectId || !Number.isSafeInteger(value.expiresAt) || value.expiresAt <= Date.now()) return false;
    const project = value.project;
    return Boolean(project && project.project_id === projectId && typeof project.title === "string"
      && Array.isArray(project.chat_threads) && ["active", "archived"].includes(project.status));
  };
  const render = (snapshot) => {
    const project = snapshot.project;
    byId("offline-title").textContent = project.title;
    byId("offline-description").textContent = project.description || "没有项目说明";
    byId("offline-saved").textContent = `快照时间：${new Date(snapshot.savedAt).toLocaleString("zh-CN")}`;
    const messages = project.chat_threads.flatMap((thread) => Array.isArray(thread.messages) ? thread.messages : [])
      .filter((message) => message && ["completed", "stopped", "failed"].includes(message.status))
      .slice(-MAX_VISIBLE_MESSAGES);
    const log = byId("offline-messages");
    log.replaceChildren();
    if (!messages.length) {
      const empty = document.createElement("p");
      empty.textContent = "这个只读快照没有可显示的终态消息。";
      log.append(empty);
    } else {
      for (const message of messages) {
        const article = document.createElement("article");
        const label = document.createElement("strong");
        label.textContent = message.role === "assistant" ? "教学 Agent" : message.role === "user" ? "学习者" : "工具";
        const body = document.createElement("p");
        body.textContent = typeof message.content === "string" ? message.content : "";
        article.append(label, body);
        log.append(article);
      }
    }
    byId("offline-status").textContent = "当前离线。以下内容来自与当前账号缓存范围绑定的只读快照；恢复联网后请重新核对服务器版本。";
    byId("offline-project").hidden = false;
  };

  void (async () => {
    try {
      if (accountTransitionPending()) {
        showFailure("账号切换尚未完成。为保护上一账号的本机数据，离线模式不会显示任何快照；请恢复联网后完成登录核对。");
        return;
      }
      const accountScope = localStorage.getItem(ACCOUNT_SCOPE_KEY) || "";
      const projectId = localStorage.getItem(ACTIVE_PROJECT_KEY) || "";
      if (!ACCOUNT_SCOPE_PATTERN.test(accountScope) || !projectId || projectId.length > 160) {
        showFailure("当前离线，且没有可验证的账号绑定快照。请恢复联网后登录。");
        return;
      }
      const snapshot = await readSnapshot(projectId);
      if (!validSnapshot(snapshot, projectId, accountScope)) {
        showFailure("当前离线。缓存已过期、损坏或不属于当前账号范围，因此不会显示。请恢复联网后登录。");
        return;
      }
      render(snapshot);
    } catch {
      showFailure("当前离线，但浏览器拒绝读取本机快照。请恢复联网后重试。");
    } finally {
      byId("offline-retry")?.addEventListener("click", () => location.reload());
      byId("offline-main")?.focus();
    }
  })();
})();
