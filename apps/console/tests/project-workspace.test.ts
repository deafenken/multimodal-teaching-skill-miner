import assert from "node:assert/strict";
import test from "node:test";

import {nextVisibleCount, projectScopedItems} from "../lib/project-workspace.ts";
import type {LearningProject} from "../lib/types.ts";

const project = {
  teaching_session_ids: ["session-owned"],
} as LearningProject;

test("project identity fence hides cached Teach handles owned by another project", () => {
  const visible = projectScopedItems([
    {session_id: "session-owned", privateValue: "owned"},
    {session_id: "session-other", privateValue: "must-not-render"},
  ], project);
  assert.deepEqual(visible, [{session_id: "session-owned", privateValue: "owned"}]);
  assert.deepEqual(projectScopedItems([{session_id: "session-owned"}], null), []);
});

test("load-more pagination reaches every item without a hidden 20/100 cap", () => {
  let visible = 0;
  while (visible < 1_237) visible = nextVisibleCount(visible, 1_237, 50);
  assert.equal(visible, 1_237);
  assert.equal(nextVisibleCount(1_237, 1_237, 50), 1_237);
});
