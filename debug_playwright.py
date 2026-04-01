from playwright.sync_api import sync_playwright

def run_cuj(page):
    page.goto("http://localhost:3000")
    page.wait_for_timeout(5000)

    # Take screenshot at the key moment
    page.screenshot(path="/home/jules/verification/screenshots/debug_on_load.png")

    # Try clicking using iframe strategy if we are in iframe
    try:
        page.get_by_text("Performance Recap").click(timeout=5000)
    except Exception as e:
        print("Could not click by text:", e)

    page.wait_for_timeout(3000)
    page.screenshot(path="/home/jules/verification/screenshots/debug_after_text_click.png")

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080}
        )
        page = context.new_page()
        try:
            run_cuj(page)
        finally:
            context.close()
            browser.close()
