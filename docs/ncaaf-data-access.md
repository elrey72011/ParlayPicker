# Check historical NCAAF access

After deployment, open **Data Maintenance â†’ NCAAF Data Access** in the sidebar and click **Check NCAAF Data Access**. The app reads your existing `CFBD_API_KEY`, `CFBDAPIKEY`, or `cfbd_api_key` secret (environment variables are also supported). Do not paste credentials into chat or commit them.

The check makes at most six requests: week 1 regular-season FBS results for each of the three prior calendar years, followed by one completed game's team statistics per year. It stops on request errors, including unauthorized, forbidden, and rate-limited responses. It only runs when clicked.

Download **NCAAF Data Access Report** and share that JSON for review. It contains fixed status labels, sample counts and game IDs, never credentials or response bodies. An accessible sample establishes that these requests worked, not full historical coverage, feature completeness, or model accuracy. The next step is auditing full historical coverage and constructing strictly pregame training features before fitting a model.

API reference: [CFBD games and team statistics](https://api.collegefootballdata.com/api/games).

After sample access succeeds, use the [historical collection workflow](ncaaf-historical-collection.md).
