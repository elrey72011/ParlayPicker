from playwright.sync_api import sync_playwright
import time

def verify_dashboard():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            print("Navigating to dashboard...")
            page.goto("http://localhost:8501", timeout=30000)

            # Wait for Streamlit to load
            page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=30000)
            print("Streamlit loaded.")

            # Allow some time for metrics to render if any
            time.sleep(5)

            # Take screenshot of the whole page
            page.screenshot(path="dashboard.png", full_page=True)
            print("Screenshot saved to dashboard.png")

            # Try to find the metric (it might not be there if no data)
            content = page.content()
            if "NBA Coverage" in content:
                print("Found 'NBA Coverage' in content.")
            else:
                print("'NBA Coverage' NOT found (expected if no data loaded).")

            if "Kalshi Matches" in content:
                print("Found 'Kalshi Matches' in content.")
            else:
                print("'Kalshi Matches' NOT found (expected if no data loaded).")

        except Exception as e:
            print(f"Error: {e}")
            page.screenshot(path="error.png")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_dashboard()
