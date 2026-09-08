# Public website connection: Netlify

This integration uploads only the validated single-page board and cache headers
from memory. It never uploads a repository directory, secrets, or raw exports.
No account, project, paid plan, or domain is created automatically by this PR.

## One-time setup

1. Create a dedicated static site/project in your Netlify account. Use a new site,
   not an unrelated existing website: publishing replaces the entire site.
2. Copy its **site/project ID** and create an access token in Netlify. Keep the
   token private; it is separate from the owner publishing password.
3. Add these top-level Streamlit secrets:

```toml
PARLAYPICKER_NETLIFY_SITE_ID = "your-site-id"
PARLAYPICKER_NETLIFY_TOKEN = "your-private-netlify-token"
```

Keep PARLAYPICKER_PUBLISH_TOKEN configured as before. Environment variables also
work for local Streamlit. Never commit tokens or paste them in chat. Check your
Netlify plan's current limits and any data-provider publication permissions
before making the board public. No free hosting or cost ceiling is promised.

## Daily publication

Run analysis → Workspace → Preview & Publish → unlock → choose optional props
and DFS → Build preview. Review the exact board. Under **Publish to public
website**, click **Verify public destination** and check the returned HTTPS URL.
Then click **Publish reviewed board publicly**. This replaces the entire site
with that reviewed board; data omitted from this publication is not carried over.

Click **Check public deployment status** after submission. The panel reports
success only when Netlify returns a ready deployment AND identifies that deploy
as the site's published deployment. A ready but unpublished deployment may need
attention in Netlify (for example, locked publishing). Browser access to the
site still needs to be checked after the first live deployment.

A submitted board cannot be sent repeatedly in the same Streamlit session.
Failed/uncertain submissions are not retried automatically. Check the Netlify
Deploys dashboard first; explicitly allow another attempt only after resolving
the previous one. Session loss removes this duplicate protection, so check the
dashboard before resubmitting after a restart. The previous public deployment
can be restored in Netlify; local rollback affects only local output.

The public HTML embeds all included picks and lineups, with no paywall. It makes
no visitor-triggered API calls. Stale/started labels use original analysis times
and the visitor clock. This is a static public board, not a billing or account
system. Custom-domain configuration and paid content remain separate work.

## Provider contract

Implementation follows Netlify's documented ZIP deploy and status APIs:
https://docs.netlify.com/api-and-cli-guides/api-guides/get-started-with-api/

Requests go only to api.netlify.com with redirects disabled. Error messages omit
provider response bodies and credentials. A deployment consists of index.html
and _headers. Verification/polling are read-only; only the public publish button
sends a deployment POST. API behavior is tested with mocks; no live account was
used during implementation.
