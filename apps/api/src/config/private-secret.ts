import {lstatSync, readFileSync, realpathSync} from "node:fs";
import {dirname, isAbsolute, resolve} from "node:path";

export class PrivateSecretConfigurationError extends Error {
  constructor(readonly code: "ambiguous" | "inline_forbidden" | "unsafe_file" | "invalid_value") {
    super("private secret configuration is invalid");
    this.name = "PrivateSecretConfigurationError";
  }
}

export interface PrivateSecretOptions {
  name: string;
  file: string | undefined;
  inline: string | undefined;
  production: boolean;
  required: boolean;
  minimumBytes?: number;
  maximumBytes?: number;
  pattern?: RegExp;
}

function validateValue(value: string, options: PrivateSecretOptions): string {
  const minimum = options.minimumBytes ?? 32;
  const maximum = options.maximumBytes ?? 4_096;
  const bytes = Buffer.byteLength(value, "utf8");
  if (
    bytes < minimum ||
    bytes > maximum ||
    /[\u0000\r\n]/.test(value) ||
    value !== value.trim() ||
    (options.pattern && !options.pattern.test(value))
  ) {
    throw new PrivateSecretConfigurationError("invalid_value");
  }
  return value;
}

function readOwnedPrivateFile(path: string, maximumBytes: number): string {
  if (!isAbsolute(path) || path.length > 512 || resolve(path) !== path) {
    throw new PrivateSecretConfigurationError("unsafe_file");
  }
  try {
    const parent = lstatSync(dirname(path));
    const file = lstatSync(path);
    const effectiveUser = typeof process.geteuid === "function" ? process.geteuid() : file.uid;
    if (
      parent.isSymbolicLink() ||
      !parent.isDirectory() ||
      parent.uid !== effectiveUser ||
      (parent.mode & 0o077) !== 0 ||
      file.isSymbolicLink() ||
      !file.isFile() ||
      file.uid !== effectiveUser ||
      (file.mode & 0o077) !== 0 ||
      file.size < 1 ||
      file.size > maximumBytes + 1 ||
      realpathSync(path) !== path
    ) {
      throw new PrivateSecretConfigurationError("unsafe_file");
    }
    const bytes = readFileSync(path);
    if (bytes.includes(0)) throw new PrivateSecretConfigurationError("unsafe_file");
    const text = bytes.toString("utf8");
    return text.endsWith("\n") ? text.slice(0, -1) : text;
  } catch (error) {
    if (error instanceof PrivateSecretConfigurationError) throw error;
    throw new PrivateSecretConfigurationError("unsafe_file");
  }
}

/**
 * Resolve one secret without ever returning its source path or value in an
 * error. Production accepts only an owned 0600 file in an owned 0700 parent;
 * development may use one inline value for local fixtures.
 */
export function privateSecret(options: PrivateSecretOptions): string | undefined {
  const file = options.file?.trim();
  const inline = options.inline;
  if (file && inline) throw new PrivateSecretConfigurationError("ambiguous");
  if (options.production && inline) {
    throw new PrivateSecretConfigurationError("inline_forbidden");
  }
  if (!file && inline === undefined) {
    if (options.required) throw new PrivateSecretConfigurationError("invalid_value");
    return undefined;
  }
  const value = file
    ? readOwnedPrivateFile(file, options.maximumBytes ?? 4_096)
    : inline!;
  return validateValue(value, options);
}
