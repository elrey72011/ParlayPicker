# Publish directly to Namecheap

This replaces manual HTML uploads with the owner-triggered Streamlit publisher.
No publisher file is uploaded to cPanel. The app uploads only the validated
public index.html, using SFTP and an atomic rename. No repository, credentials,
raw exports, .htaccess, or main-site files are uploaded. Netlify remains the
default until you explicitly select sftp.

## One-time setup for this migration

1. Merge the publisher PR and let Streamlit install the updated dependencies.
2. Finish DNS and SSL for https://picks.cmsvconsulting.com. The existing manually
   uploaded index.html must load over HTTPS without a certificate warning.
3. Namecheap shared hosting uses SFTP port 21098. Confirm SFTP access for your
   cPanel account. Ask Namecheap support for the server's SSH host-key SHA256
   fingerprint, including which key type it identifies. A fingerprint is public;
   your account password is private. Do not trust an unverified key scan alone.
4. Add these top-level values in Streamlit Manage app > Settings > Secrets:

```toml
PARLAYPICKER_PUBLIC_PROVIDER = "sftp"
PARLAYPICKER_PUBLIC_URL = "https://picks.cmsvconsulting.com"
PARLAYPICKER_SFTP_HOST = "premium157.web-hosting.com"
PARLAYPICKER_SFTP_PORT = "21098"
PARLAYPICKER_SFTP_USER = "cmsvorjp"
PARLAYPICKER_SFTP_DIRECTORY = "/home/cmsvorjp/picks.cmsvconsulting.com"
PARLAYPICKER_SFTP_PASSWORD = "YOUR_CPANEL_PASSWORD"
PARLAYPICKER_SFTP_HOST_KEY_SHA256 = "SHA256:VERIFIED_SERVER_FINGERPRINT"
```

Keep your existing PARLAYPICKER_PUBLISH_TOKEN, Drive configuration and
PARLAYPICKER_NETLIFY_SITE_ID. The old site ID remains the stable history namespace;
changing it would hide prior results. Keep the same GitHub Actions site-ID variable
so scheduled grading continues to use that history. Netlify credentials are not
used for new SFTP uploads. The old Netlify token can still recover an unconfirmed
legacy Netlify deployment during Restore history; retain it until those are resolved.
Never commit real credentials or paste passwords into chat.

The directory is intentionally restricted to /home/USERNAME/SUBDOMAIN and must
not be a symlink. Other layouts require a separately reviewed configuration change.
If support provides multiple host keys, configure the fingerprint of the key
negotiated by the SSH client; a mismatch fails closed without sending credentials.

## Publish

Restore public history, build and review the preview, then choose Verify public
destination under Publish to Namecheap. Verification reads the SFTP directory and
checks the HTTPS site; it does not upload. Click Publish reviewed board publicly,
then Check public deployment status. Success requires the public HTTPS page's
SHA256 to match the exact uploaded HTML; only then is the archived package confirmed
in Drive. Publishing remains manual. Visitors make no sports API calls.

If upload or verification fails, use Check public deployment status before allowing
another attempt. After a Streamlit restart, Restore public history can recover an
unconfirmed SFTP receipt if its exact HTML is still live. It never assumes an old,
replaced page was published. Cache or proxy HTML rewriting causes a mismatch; serve
the original index.html without HTML transformation. HTTPS redirects are not followed.

Upload uses a unique temporary file in the subdomain folder and POSIX atomic rename.
If the server does not support atomic replacement, it fails without deleting the
old index.html. There is no automatic rollback; retain a downloaded previous HTML
for manual restoration through cPanel File Manager. Existing main-site files,
SSL validation directories, and server configuration are left untouched.

No live SFTP account was used in implementation tests. The first real connection
must verify hosting access, its host key, atomic rename support and HTTPS delivery.
This uses your existing hosting allocation; account limits and renewals still apply.

Sources:
- https://www.namecheap.com/support/knowledgebase/article.aspx/131/89/do-you-provide-ssh-if-yes-under-what-conditions/
- https://www.namecheap.com/support/knowledgebase/article.aspx/188/205/how-to-access-an-account-via-ftp/
- https://docs.paramiko.org/en/stable/api/client.html
- https://docs.paramiko.org/en/stable/api/sftp.html
