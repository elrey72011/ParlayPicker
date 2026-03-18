from playwright.sync_api import sync_playwright
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(record_video_dir=".", viewport={'width': 1280, 'height': 720})
        page = context.new_page()

        page.goto("http://localhost:8501")
        page.wait_for_timeout(5000)

        # Click the sidebar expander button if it's hidden
        try:
            page.locator('button[data-testid="collapsedControl"]').click()
            page.wait_for_timeout(1000)
        except Exception:
            pass

        page.screenshot(path="ui_screenshot2.png")

        # Look for the Advanced Settings expander
        advanced_settings = page.locator('text="Advanced Settings"')

        if advanced_settings.count() > 0:
            print("Advanced Settings expander found!")
            advanced_settings.first.click()
            page.wait_for_timeout(1000)

            show_all_games = page.locator('text="SHOW_ALL_GAMES (Bypass Edge Filter)"')
            if show_all_games.count() > 0:
                print("SHOW_ALL_GAMES toggle found!")
                show_all_games.first.click()
                page.wait_for_timeout(1000)
            else:
                print("SHOW_ALL_GAMES toggle NOT found.")
        else:
            print("Advanced Settings expander NOT found.")

        page.screenshot(path="ui_screenshot3.png")

        context.close()
        browser.close()

if __name__ == "__main__":
    run()
