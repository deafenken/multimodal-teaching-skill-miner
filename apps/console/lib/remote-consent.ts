/** Browser-local v1 timestamps were never server authority. */
export const legacyRemoteConsentStorageKey = "teachlab.remote-consent.v1";

export function invalidateLegacyRemoteConsent(storage: Pick<Storage, "getItem" | "removeItem"> = window.localStorage): boolean {
  const existed = storage.getItem(legacyRemoteConsentStorageKey) !== null;
  storage.removeItem(legacyRemoteConsentStorageKey);
  return existed;
}
