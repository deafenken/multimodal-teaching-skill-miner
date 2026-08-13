"""Private, rotation-aware Ed25519 keyring for curriculum seals.

This keyring lives only inside one confined gateway worker root.  Browser and
worker HTTP request bodies never carry signing material.  The registry is
authenticated with a stable scope key, atomically replaced, and exposes only
non-revoked public keys to runtime verification.  Consequently revoking an old
key makes every seal made solely by that key fail closed immediately.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Iterator, Mapping

try:  # pragma: no cover - production is POSIX.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .io_utils import ensure_private_directory


CURRICULUM_SIGNING_KEYRING_SCHEMA = (
    "teaching_skill_miner.curriculum_signing_keyring.v1"
)
_KEY_ID = re.compile(r"^curriculum-key-[0-9a-f]{24}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_MAX_KEYS = 32
_MAX_BYTES = 256 * 1024


class CurriculumSigningKeyringError(RuntimeError):
    """Raised when private curriculum signing material is not trustworthy."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CurriculumSigningKeyringError(
            "curriculum signing keyring must be canonical JSON"
        ) from exc


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class CurriculumSigningKeyring:
    """Authenticated private key registry with explicit rotate/revoke hooks."""

    def __init__(self, path: str | Path, *, integrity_key: bytes) -> None:
        if not isinstance(integrity_key, bytes) or len(integrity_key) != 32:
            raise CurriculumSigningKeyringError(
                "curriculum signing keyring integrity key is invalid"
            )
        candidate = Path(path).expanduser()
        if candidate.exists() and candidate.is_symlink():
            raise CurriculumSigningKeyringError(
                "curriculum signing keyring cannot be a symlink"
            )
        self.root = ensure_private_directory(candidate.parent).resolve()
        self.path = self.root / candidate.name
        self.lock_path = self.root / f".{candidate.name}.lock"
        root_key = bytes(integrity_key)
        self._integrity_key = hmac.new(
            root_key, b"teachlab-curriculum-keyring-integrity-v1", sha256
        ).digest()
        self._aead_key = hmac.new(
            root_key, b"teachlab-curriculum-keyring-private-aead-v1", sha256
        ).digest()
        self._scope_binding_sha256 = hmac.new(
            root_key, b"teachlab-curriculum-keyring-scope-binding-v1", sha256
        ).hexdigest()
        self._lock = threading.RLock()
        with self._guard():
            if not self.path.exists():
                state = self._empty()
                state = self._add_key(state, activate=True)
                self._write(state)
            else:
                self._read()

    @contextmanager
    def _guard(self) -> Iterator[None]:
        with self._lock:
            descriptor = os.open(
                self.lock_path,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _empty(self) -> dict[str, Any]:
        return {
            "schema": CURRICULUM_SIGNING_KEYRING_SCHEMA,
            "version": 1,
            "scope_binding_sha256": self._scope_binding_sha256,
            "active_key_id": None,
            "keys": {},
            "integrity_hmac_sha256": "",
        }

    def _seal(self, state: Mapping[str, Any]) -> dict[str, Any]:
        output = deepcopy(dict(state))
        output.pop("integrity_hmac_sha256", None)
        output["integrity_hmac_sha256"] = hmac.new(
            self._integrity_key, _canonical(output), sha256
        ).hexdigest()
        return output

    def _add_key(self, state: Mapping[str, Any], *, activate: bool) -> dict[str, Any]:
        output = deepcopy(dict(state))
        keys = output["keys"]
        if len(keys) >= _MAX_KEYS:
            raise CurriculumSigningKeyringError(
                "curriculum signing key capacity is exhausted"
            )
        private_key = Ed25519PrivateKey.generate()
        private_pem = private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        public_raw = private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        key_id = "curriculum-key-" + sha256(public_raw).hexdigest()[:24]
        if key_id in keys:  # cryptographically negligible, still fail closed.
            raise CurriculumSigningKeyringError(
                "curriculum signing key identity collision"
            )
        public_key_base64 = base64.b64encode(public_raw).decode("ascii")
        nonce = os.urandom(12)
        ciphertext = AESGCM(self._aead_key).encrypt(
            nonce,
            private_pem,
            self._private_key_aad(
                key_id=key_id,
                public_key_base64=public_key_base64,
            ),
        )
        keys[key_id] = {
            "public_key_base64": public_key_base64,
            "private_key_nonce_base64": base64.b64encode(nonce).decode("ascii"),
            "private_key_ciphertext_base64": base64.b64encode(ciphertext).decode(
                "ascii"
            ),
            "status": "active" if activate else "verification_only",
            "created_at_utc": _utc(),
            "revoked_at_utc": None,
        }
        if activate:
            previous = output.get("active_key_id")
            if isinstance(previous, str) and previous in keys:
                keys[previous]["status"] = "verification_only"
            output["active_key_id"] = key_id
        return output

    def _private_key_aad(
        self, *, key_id: str, public_key_base64: str
    ) -> bytes:
        return _canonical(
            {
                "schema": CURRICULUM_SIGNING_KEYRING_SCHEMA,
                "key_id": key_id,
                "scope_binding_sha256": self._scope_binding_sha256,
                "public_key_base64": public_key_base64,
            }
        )

    def _decrypt_private_key(self, key_id: str, raw: Mapping[str, Any]) -> bytes:
        try:
            nonce = base64.b64decode(
                str(raw["private_key_nonce_base64"]), validate=True
            )
            ciphertext = base64.b64decode(
                str(raw["private_key_ciphertext_base64"]), validate=True
            )
            if len(nonce) != 12 or len(ciphertext) < 17:
                raise ValueError("invalid AES-GCM envelope")
            return AESGCM(self._aead_key).decrypt(
                nonce,
                ciphertext,
                self._private_key_aad(
                    key_id=key_id,
                    public_key_base64=str(raw["public_key_base64"]),
                ),
            )
        except (InvalidTag, KeyError, TypeError, ValueError) as exc:
            raise CurriculumSigningKeyringError(
                "curriculum signing private key envelope is invalid"
            ) from exc

    def _validate(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping) or set(value) != {
            "schema",
            "version",
            "scope_binding_sha256",
            "active_key_id",
            "keys",
            "integrity_hmac_sha256",
        }:
            raise CurriculumSigningKeyringError(
                "curriculum signing keyring shape is invalid"
            )
        if (
            value.get("schema") != CURRICULUM_SIGNING_KEYRING_SCHEMA
            or value.get("version") != 1
            or value.get("scope_binding_sha256") != self._scope_binding_sha256
            or not isinstance(value.get("keys"), Mapping)
            or not 1 <= len(value["keys"]) <= _MAX_KEYS
        ):
            raise CurriculumSigningKeyringError(
                "curriculum signing keyring boundary is invalid"
            )
        material = deepcopy(dict(value))
        declared = material.pop("integrity_hmac_sha256", None)
        if (
            not isinstance(declared, str)
            or not hmac.compare_digest(
                declared,
                hmac.new(self._integrity_key, _canonical(material), sha256).hexdigest(),
            )
        ):
            raise CurriculumSigningKeyringError(
                "curriculum signing keyring integrity failed"
            )
        active = value.get("active_key_id")
        if not isinstance(active, str) or _KEY_ID.fullmatch(active) is None:
            raise CurriculumSigningKeyringError(
                "curriculum signing active key is invalid"
            )
        for key_id, raw in value["keys"].items():
            if (
                not isinstance(key_id, str)
                or _KEY_ID.fullmatch(key_id) is None
                or not isinstance(raw, Mapping)
                or set(raw)
                != {
                    "public_key_base64",
                    "private_key_nonce_base64",
                    "private_key_ciphertext_base64",
                    "status",
                    "created_at_utc",
                    "revoked_at_utc",
                }
                or raw.get("status")
                not in {"active", "verification_only", "revoked"}
                or not isinstance(raw.get("created_at_utc"), str)
                or _UTC.fullmatch(str(raw["created_at_utc"])) is None
            ):
                raise CurriculumSigningKeyringError(
                    "curriculum signing key record is invalid"
                )
            if raw["status"] == "revoked":
                if (
                    not isinstance(raw.get("revoked_at_utc"), str)
                    or _UTC.fullmatch(str(raw["revoked_at_utc"])) is None
                ):
                    raise CurriculumSigningKeyringError(
                        "revoked curriculum signing key timestamp is invalid"
                    )
            elif raw.get("revoked_at_utc") is not None:
                raise CurriculumSigningKeyringError(
                    "active curriculum signing key cannot carry revocation time"
                )
            try:
                public_raw = base64.b64decode(
                    str(raw["public_key_base64"]), validate=True
                )
                private_pem = self._decrypt_private_key(key_id, raw)
                private_key = serialization.load_pem_private_key(
                    private_pem, password=None
                )
            except (ValueError, TypeError) as exc:
                raise CurriculumSigningKeyringError(
                    "curriculum signing key material is invalid"
                ) from exc
            if (
                len(public_raw) != 32
                or not isinstance(private_key, Ed25519PrivateKey)
                or private_key.public_key().public_bytes(
                    serialization.Encoding.Raw,
                    serialization.PublicFormat.Raw,
                )
                != public_raw
                or key_id
                != "curriculum-key-" + sha256(public_raw).hexdigest()[:24]
            ):
                raise CurriculumSigningKeyringError(
                    "curriculum signing public/private binding is invalid"
                )
        if value["keys"][active]["status"] != "active" or sum(
            raw["status"] == "active" for raw in value["keys"].values()
        ) != 1:
            raise CurriculumSigningKeyringError(
                "curriculum signing keyring must have one active key"
            )
        if len(_canonical(value)) > _MAX_BYTES:
            raise CurriculumSigningKeyringError(
                "curriculum signing keyring exceeds its byte limit"
            )
        return deepcopy(dict(value))

    def _read(self) -> dict[str, Any]:
        if self.path.is_symlink() or not self.path.is_file():
            raise CurriculumSigningKeyringError(
                "curriculum signing keyring path is unsafe"
            )
        try:
            raw = self.path.read_bytes()
            if len(raw) > _MAX_BYTES:
                raise CurriculumSigningKeyringError(
                    "curriculum signing keyring exceeds its byte limit"
                )
            value = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CurriculumSigningKeyringError(
                "curriculum signing keyring cannot be read"
            ) from exc
        return self._validate(value)

    def _write(self, state: Mapping[str, Any]) -> None:
        sealed = self._seal(state)
        candidate = self._validate(sealed)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.root
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", buffering=0) as handle:
                handle.write(_canonical(candidate) + b"\n")
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def active_signing_material(self) -> tuple[str, bytes]:
        with self._guard():
            state = self._read()
        key_id = str(state["active_key_id"])
        private_pem = self._decrypt_private_key(key_id, state["keys"][key_id])
        return key_id, private_pem

    def trusted_public_keys(self) -> dict[str, bytes]:
        with self.trusted_public_keys_lease() as keys:
            return dict(keys)

    def all_public_keys(self) -> dict[str, bytes]:
        """Return the private-runtime audit registry, including revoked keys.

        Revoked keys must never authorize a lesson again, but their public
        bytes remain necessary to verify the immutable signatures of historic
        review/seal events.  This server-only method is deliberately separate
        from both :meth:`trusted_public_keys` and the credential-free bootstrap
        projection.
        """

        with self._guard():
            state = self._read()
            return {
                key_id: base64.b64decode(raw["public_key_base64"], validate=True)
                for key_id, raw in state["keys"].items()
            }

    @contextmanager
    def trusted_public_keys_lease(self) -> Iterator[dict[str, bytes]]:
        """Hold key lifecycle fencing through a dependent grading commit."""

        with self._guard():
            state = self._read()
            yield {
                key_id: base64.b64decode(raw["public_key_base64"], validate=True)
                for key_id, raw in state["keys"].items()
                if raw["status"] in {"active", "verification_only"}
            }

    def public_status(self) -> dict[str, Any]:
        """Return lifecycle metadata without public bytes or private envelopes."""

        with self._guard():
            state = self._read()
        return {
            "schema": "teaching_skill_miner.curriculum_signing_key_status.v1",
            "active_key_id": state["active_key_id"],
            "keys": [
                {
                    "key_id": key_id,
                    "status": raw["status"],
                    "created_at_utc": raw["created_at_utc"],
                    "revoked_at_utc": raw["revoked_at_utc"],
                    "public_key_sha256": sha256(
                        base64.b64decode(raw["public_key_base64"], validate=True)
                    ).hexdigest(),
                }
                for key_id, raw in sorted(state["keys"].items())
            ],
            "private_key_material_exposed": False,
        }

    def rotate(self) -> str:
        """Generate and activate a new key while retaining old verification."""

        with self._guard():
            state = self._add_key(self._read(), activate=True)
            self._write(state)
            return str(state["active_key_id"])

    def revoke(self, key_id: str) -> None:
        """Remove a non-active key from the runtime trust registry."""

        if _KEY_ID.fullmatch(key_id) is None:
            raise CurriculumSigningKeyringError(
                "curriculum signing key ID is invalid"
            )
        with self._guard():
            state = self._read()
            if key_id == state["active_key_id"]:
                raise CurriculumSigningKeyringError(
                    "rotate before revoking the active curriculum signing key"
                )
            key = state["keys"].get(key_id)
            if key is None:
                raise CurriculumSigningKeyringError(
                    "curriculum signing key was not found"
                )
            if key["status"] == "revoked":
                return
            key["status"] = "revoked"
            key["revoked_at_utc"] = _utc()
            self._write(state)


__all__ = [
    "CURRICULUM_SIGNING_KEYRING_SCHEMA",
    "CurriculumSigningKeyring",
    "CurriculumSigningKeyringError",
]
