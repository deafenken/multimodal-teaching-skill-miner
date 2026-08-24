# Legacy local data boundary

This directory is intentionally preserved during the Agent Harness migration because it
may contain operator-owned local data from earlier versions.

Agent Harness does not read, migrate, upload or delete anything here. The directory is
ignored by the active package and must not be committed to source control. Remove it only
after an operator has independently backed up and reviewed its contents.
