import os
from playwright.sync_api import sync_playwright

def run_cuj(page):
    page.goto("http://localhost:8502")
    page.wait_for_timeout(2000) # wait for streamlit to load

    # We should see the prior day performance header
    page.wait_for_selector("text=Prior Day Performance", timeout=5000)
    page.wait_for_timeout(1000)

    # Let's interact with the data editor to grade some picks
    # We have 2 mocked games:
    # 1. Lakers ML (Actionable) -> Let's give Lakers the win (Score 120-100)
    # 2. Heat +5.5 (Actionable) -> Let's give Heat a loss but cover (Heat 100, Celtics 105 -> Margin 5 -> Cover!)

    # Fill in actual_home_score and actual_away_score using keyboard
    # In streamlit data editor, we can tab through cells
    # The first editable cell should be Home Score for row 1
    page.keyboard.press("Tab") # usually focuses first input/element
    # Need to navigate to the first editable cell, or we can just double click if we know the coordinates,
    # but let's just use the manual dropdown "Outcome" to grade them to test the logic directly if possible.
    # Actually our mocked data doesn't have actual scores populated yet.

    # Let's test the UI rendering of the labels we added:
    page.wait_for_selector("text=Total Net Profit - Actionable (Units)", timeout=5000)
    page.wait_for_selector("text=Total Net Profit - All Picks (Units)", timeout=5000)
    page.wait_for_timeout(1000)

    # Take screenshot
    page.screenshot(path="/home/jules/verification/screenshots/verification.png")
    page.wait_for_timeout(1000)

if __name__ == "__main__":
    os.makedirs("/home/jules/verification/videos", exist_ok=True)
    os.makedirs("/home/jules/verification/screenshots", exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            record_video_dir="/home/jules/verification/videos"
        )
        page = context.new_page()
        try:
            run_cuj(page)
        finally:
            context.close()
            browser.close()
