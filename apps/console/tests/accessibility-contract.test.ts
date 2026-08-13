import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";

import {isRovingMenuKey, nextRovingMenuIndex} from "../lib/accessibility.ts";

const conversation = readFileSync(new URL("../components/workbench/conversation-pane.tsx", import.meta.url), "utf8");
const inspector = readFileSync(new URL("../components/workbench/inspector-drawer.tsx", import.meta.url), "utf8");
const projectSwitcher = readFileSync(new URL("../components/workbench/project-switcher.tsx", import.meta.url), "utf8");

function assertSourceContract(source: string, required: string[]) {
  for (const token of required) assert.ok(source.includes(token), `missing accessibility contract token: ${token}`);
}

test("roving menu arithmetic wraps and supports first/last keys", () => {
  assert.equal(nextRovingMenuIndex(0, 3, "ArrowUp"), 2);
  assert.equal(nextRovingMenuIndex(2, 3, "ArrowDown"), 0);
  assert.equal(nextRovingMenuIndex(1, 3, "Home"), 0);
  assert.equal(nextRovingMenuIndex(1, 3, "End"), 2);
  assert.equal(nextRovingMenuIndex(0, 0, "ArrowDown"), -1);
  assert.equal(isRovingMenuKey("Home"), true);
  assert.equal(isRovingMenuKey("Tab"), false);
});

test("conversation surface has named input, polite log, and owned keyboard menus", () => {
  assertSourceContract(conversation, [
    'role="log"',
    'aria-live="polite"',
    'aria-relevant="additions text"',
    'htmlFor="teachlab-composer-input"',
    'id="teachlab-composer-input"',
    'id="teachlab-command-options"',
    'role="listbox"',
    'role="option"',
    'aria-activedescendant=',
    'aria-controls="teachlab-control-menu"',
    'role="menuitemradio"',
    "nextRovingMenuIndex",
    "controlMenuTriggerRef.current?.focus()",
  ]);
  assert.match(conversation, /语音输入不可用：未配置受信任的本地 ASR/);
  assert.doesNotMatch(conversation, /SpeechRecognition|webkitSpeechRecognition|recognition\.start/);
});

test("project menu and mobile inspector deterministically restore and contain focus", () => {
  assertSourceContract(projectSwitcher, [
    'aria-controls="teachlab-project-menu"',
    'id="teachlab-project-menu"',
    'role="menu"',
    'role="menuitem"',
    'data-roving-menuitem="true"',
    "nextRovingMenuIndex",
    "triggerRef.current?.focus()",
  ]);
  assertSourceContract(inspector, [
    'role={isMobile ? "dialog" : "complementary"}',
    "aria-modal={isMobile ? true : undefined}",
    'event.key !== "Tab"',
    "document.activeElement === first",
    "document.activeElement === last",
    "previous?.focus?.()",
  ]);
});
