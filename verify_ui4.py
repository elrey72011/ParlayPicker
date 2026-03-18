from playwright.sync_api import sync_playwright

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

        # Use an exact class or test-id or text locator that is more robust
        advanced_settings_div = page.locator('div[data-testid="stExpanderDetails"]').locator('..')

        # Let's try getting all text that says Advanced Settings
        advanced_settings = page.locator('text="Advanced Settings"')
        if advanced_settings.count() > 0:
            print("Advanced Settings found!")
            # Try to expand it by clicking the parent div or summary
            try:
                page.locator('summary:has-text("Advanced Settings")').click()
                print("Clicked summary element!")
                page.wait_for_timeout(1000)
            except Exception as e:
                print(f"Failed to click summary: {e}")

            show_all_games = page.locator('text="SHOW_ALL_GAMES (Bypass Edge Filter)"')
            if show_all_games.count() > 0:
                print("SHOW_ALL_GAMES toggle found!")
                # Streamlit checkboxes are labels containing input
                try:
                    page.locator('label:has-text("SHOW_ALL_GAMES (Bypass Edge Filter)")').click()
                    print("Clicked toggle!")
                    page.wait_for_timeout(1000)
                except Exception as e:
                    print(f"Failed to click toggle: {e}")
            else:
                print("SHOW_ALL_GAMES toggle NOT found.")
        else:
            print("Advanced Settings text NOT found.")

        page.screenshot(path="ui_screenshot6.png")

        context.close()
        browser.close()

if __name__ == "__main__":
    run()
