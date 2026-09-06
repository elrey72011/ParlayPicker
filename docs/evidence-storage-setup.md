# Google Workspace Shared Drive evidence storage

The app keeps its local SQLite cache and backs up original evidence records to a
dedicated folder in your existing Shared Drive. It automatically restores records
when the local cache is replaced. No AWS account or S3 bucket is needed.

## 1. Create the Google identity

1. Open [Google Cloud Console](https://console.cloud.google.com/) and create or
   select a project for ParlayPicker. Enable **Google Drive API**.
2. In **IAM & Admin → Service Accounts**, create a dedicated service account
   named `parlaypicker-evidence`. It needs no project-wide IAM role for this task.
3. Open that service account, choose **Keys → Add key → Create new key → JSON**,
   and download its key securely. If your organization prohibits service-account
   keys, ask your Workspace/Cloud administrator for an approved authentication
   method; do not change that organization policy just for this app.
4. Copy the service account's email address. Do not paste its private key into
   chat or commit its JSON file to the repository.

Service accounts cannot own personal My Drive files; they can write into Shared
Drives owned by your organization. See [Google's Shared Drive documentation](https://developers.google.com/workspace/drive/api/guides/about-shareddrives).

## 2. Grant access to one evidence folder

1. In your **Google Workspace Shared Drive**, create a folder named
   **ParlayPicker Evidence**.
2. Share that folder with the service-account email, granting permission to add
   files (Contributor or the corresponding folder-level role). Your Workspace
   administrator may need to allow that identity or add it to the Shared Drive
   if folder-level sharing is restricted. Do not make the folder public.
3. Open the folder and copy its ID: the part after `/folders/` in its URL. Use the
   ID only, without query parameters. A shared folder inside personal My Drive
   is not a Workspace Shared Drive and will be rejected by the app.

The service account uses the Drive API scope to access resources shared with that
identity. It does not impersonate your personal Google account. The application
implements creation and reading only; it never updates or deletes evidence files.
Other folder editors can still alter files, so this is not regulatory WORM storage.

## 3. Add Streamlit secrets

In the Streamlit app's **Settings → Secrets**, insert the following at the very
TOP of the existing TOML, before any `[section]` headers. Keep existing secrets.
Replace the folder placeholder. Paste the entire downloaded service-account JSON
between the triple single quotes, replacing the placeholder line:

```toml
PARLAYPICKER_DRIVE_FOLDER_ID = "YOUR_SHARED_DRIVE_FOLDER_ID"
PARLAYPICKER_DRIVE_PREFIX = "parlaypicker/evidence-v1"
PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT = '''
PASTE_THE_COMPLETE_SERVICE_ACCOUNT_JSON_HERE
'''
```

The TOML literal string preserves the JSON's escaped private-key newlines. Do not
convert those escapes into manually edited key material. Root-level secrets are
available as environment variables, which this integration reads. See
[Streamlit secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).

No Drive desktop sync application or local folder setting is required. Google
Drive storage and API quotas apply. Credentials remain in Streamlit secrets;
status downloads contain no credentials.

## 4. Verify the deployed connection

1. Merge/deploy the storage-support PR and configure the secrets above.
2. Run Master Analysis before the intended games start. Under **Prediction
   Evidence Status**, confirm the local store is healthy and **Remote backup:
   synced**. Save the status JSON and candidate/final CSV exports.
3. Open the Shared Drive folder. It should contain JSON records for model bundles,
   snapshots and runtime identities. Score-revision records appear after grading.
   Slashes in the record names are naming prefixes, not nested Drive folders.
4. Only after backup is confirmed, reboot/redeploy the app. Before running new
   analysis, open evidence status and compare the latest snapshot ID. A fresh
   filesystem should show restored snapshots with the original IDs. A reboot
   that preserves the cache is not a fresh-filesystem restoration test.
5. After settlement, grade saved decisions through Performance Recap. Confirm the
   score revision count increases and the remote status returns to `synced`.

Automatic status deliberately keeps `durability_across_redeployment_verified`
false: the app cannot independently prove the hosting platform replaced its
filesystem. Matching original IDs after an observed replacement are the acceptance
check. Tests use a fake remote service and a fresh local directory; real Drive
permissions and persistence must still be verified after configuration.

## Failures and operating limits

- **not_configured**: the folder ID is missing from root-level secrets.
- **error / backup pending**: correct the credentials, folder permissions, quota
  or network issue, then click **Restore and sync evidence storage**. Local
  records survive an upload failure; do not redeploy unbacked evidence.
- Startup restoration must succeed before a new remotely registered model freeze
  is created. An unavailable backend can therefore block evidence initialization;
  the analysis UI reports that failure instead of claiming successful capture.
- Drive allows duplicate filenames. Identical copies from uncertain upload retries
  are accepted; conflicting contents fail verification. Keep one active deployment
  writing this folder. Unlike S3, Drive has no conditional creation by filename;
  conflicting concurrent model freezes are rejected, not silently selected.
- Records use SHA-256 integrity checks and are verified by downloading after
  creation. Restoration merges transactionally and refuses conflicting local data.
- Existing downloaded CSVs preserve their reported fields but cannot reconstruct
  missing original inputs or model manifests. Evidence lost before configuration
  is not recoverable from the new folder. Keep your current exports.
- The initial implementation scans the folder and verifies local objects during
  synchronization. For a large history, incremental replication will be needed.
  No background scheduler is installed: capture, grading, and the retry button
  trigger synchronization.

## Command line

With the same three environment variables configured:

```powershell
python scripts/prediction_evidence.py backup
python scripts/prediction_evidence.py restore
python scripts/prediction_evidence.py status
```

All accept `--database PATH`. Restore to a new database path for an independent
recovery check. Backup/restore commands fail rather than report success when the
backend is unconfigured or unavailable. The Drive API implementation follows
[Google's upload documentation](https://developers.google.com/workspace/drive/api/guides/manage-uploads).


## Service-account JSON parsing error

`configured: true` means the folder setting was detected; it does not prove that
credentials parsed or that Drive permissions succeeded. If status reports a
service-account JSON error, replace the complete secret using the original JSON
file and the triple **single** quote example above. Triple double quotes process
JSON backslash escapes as TOML escapes and can turn the private-key line breaks
into invalid JSON. Do not wrap the entire JSON object in another JSON string.
Save the secrets, then click **Restore and sync evidence storage**. An empty new
folder may restore zero snapshots; run a new pregame analysis and require remote
status `synced` to verify the first backup. Never send the private key in chat.
