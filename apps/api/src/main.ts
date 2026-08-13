import "reflect-metadata";
import "dotenv/config";

import {AppConfigService} from "./config/app-config.service";
import {createApplication} from "./create-application";

async function bootstrap(): Promise<void> {
  const app = await createApplication();
  const config = app.get(AppConfigService);
  await app.listen(config.port, config.host);
}

void bootstrap().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : "Unknown startup failure";
  process.stderr.write(`TeachLab API refused startup: ${message}\n`);
  process.exitCode = 1;
});
