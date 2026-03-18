from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("http://localhost:8501", wait_until="networkidle")

        page.wait_for_timeout(2000)

        # Check if the "Advanced Settings" expander exists
        advanced_settings = page.locator('text="Advanced Settings"')

        if advanced_settings.count() > 0:
            print("Advanced Settings expander found!")
            advanced_settings.click()
            page.wait_for_timeout(1000)

            show_all_games = page.locator('text="SHOW_ALL_GAMES (Bypass Edge Filter)"')
            if show_all_games.count() > 0:
                print("SHOW_ALL_GAMES toggle found!")
            else:
                print("SHOW_ALL_GAMES toggle NOT found.")
        else:
            print("Advanced Settings expander NOT found.")

        page.screenshot(path="ui_screenshot.png")
        browser.close()

if __name__ == "__main__":
    run()
