# Public website navigation

The saved board has shared ParlayPicker navigation: Picks, Player Props, Parlays, Results, DraftKings DFS and How Picks Work. Section links use URL fragments so visitors can bookmark a section and return from the methodology page. Browser Back/Forward restores the section. Daily publishing retains this navigation through the board template.

For Namecheap, merge this change, build a fresh preview in Streamlit and publish it normally. Upload and extract the separately prepared `how-picks-work-navigation.zip` in the subdomain document root, replacing `how-picks-work/index.html`. The ZIP contains only that methodology page and sets file permissions to 0644. Preserve the root index.html containing the daily board.

To regenerate the methodology upload from the original standalone visualization export:

    python scripts/prepare_methodology_page.py PATH_TO_EXPORT outputs/site-navigation

The wrapper adds navigation outside the original iframe; its sandbox, encoded content and CSP are preserved. The methodology remains a snapshot of the supplied visualization. No account system, payment flow, live visitor API requests or new prediction logic is introduced.
