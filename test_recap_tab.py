from playwright.sync_api import sync_playwright

def run_cuj(page):
    page.goto("http://localhost:3000")
    page.wait_for_timeout(3000)

    # We do NOT need to click 'Run Master Analysis' to see the tabs, they are rendered unconditionally
    # since `tab1, tab2, tab3, tab_performance... = st.tabs(...)` is at the module level.
    # We click the Performance Recap tab immediately.
    page.get_by_role("tab", name="Performance Recap").click()
    page.wait_for_timeout(4000)

    # Take screenshot at the key moment
    page.screenshot(path="/home/jules/verification/screenshots/verification.png")
    page.wait_for_timeout(3000)

if __name__ == "__main__":
    import os
    os.makedirs("/home/jules/verification/screenshots", exist_ok=True)
    os.makedirs("/home/jules/verification/videos", exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            record_video_dir="/home/jules/verification/videos",
            viewport={"width": 1920, "height": 1080}
        )
        page = context.new_page()
        try:
            run_cuj(page)
        finally:
            context.close()
            browser.close()
