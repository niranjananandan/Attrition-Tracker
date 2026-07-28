import asyncio
from playwright.async_api import async_playwright

async def run_e2e_test():
    async with async_playwright() as p:
        
        # 1. Mobile Emulation: Grab preset configurations for an iPhone 13
        iphone_13 = p.devices['iPhone 13']
        
        # 4. Visual Feedback: Launch in headed mode (headless=False)
        # slow_mo=500 adds a 500ms delay between all Playwright operations automatically
        print("Launching Chromium browser in mobile emulation mode...")
        browser = await p.chromium.launch(headless=False, slow_mo=500)
        
        # Create a new browser context with the iPhone 13 viewport and user-agent
        context = await browser.new_context(**iphone_13)
        page = await context.new_page()

        # 2. Auto-Navigation
        print("Navigating to https://attrition-tracker.onrender.com ...")
        # Wait until there are no more than 2 network connections for at least 500 ms
        await page.goto('https://attrition-tracker.onrender.com', wait_until='networkidle')
        
        # Streamlit apps often have a secondary load state. We wait a few extra seconds to ensure the DOM is ready.
        print("Waiting for Streamlit app to fully render...")
        await asyncio.sleep(3) 

        # 3a. Find and click the Sidebar toggle button
        print("Locating and clicking the Sidebar toggle button...")
        # We target the exact data-testid that you styled in your CSS
        sidebar_toggle = page.locator('[data-testid="collapsedControl"]')
        
        # Wait for the button to become visible and clickable
        await sidebar_toggle.wait_for(state='visible', timeout=10000)
        await sidebar_toggle.click()
        
        # Pause to let you watch the sidebar slide out
        await asyncio.sleep(2)
        
        # Close the sidebar by clicking on the overlay or clicking a close button if applicable
        # (Assuming you want to scroll the main page next. If the sidebar blocks scrolling, we dismiss it.)
        try:
            close_btn = page.locator('[data-testid="stSidebar"] button').first
            await close_btn.click()
            await asyncio.sleep(1)
        except Exception:
            pass # Ignore if there's no specific close button

        # 3b. Simulate a user scrolling down the page slowly
        print("Simulating user scroll behavior...")
        for _ in range(5):
            # Scroll down by 400 pixels at a time
            await page.mouse.wheel(0, 400)
            await asyncio.sleep(0.8) # Small pause between scrolls
            
        print("Scrolling back up to find login buttons...")
        await page.mouse.wheel(0, -2000)
        await asyncio.sleep(1)

        # 3c. Click the "Google Login" or "Secure Portal" button
        print("Attempting to initiate the auth flow...")
        
        # Playwright pseudo-selectors :has-text() are case-insensitive and look for matching substrings
        login_button = page.locator('button:has-text("Google"), button:has-text("Login"), a:has-text("Google")').first
        
        try:
            await login_button.wait_for(state='visible', timeout=5000)
            print("Login button found! Clicking it...")
            await login_button.click()
            
            # Wait a few seconds to let you see the redirect happen
            await asyncio.sleep(4)
        except Exception:
            print("Could not find a button with text 'Google' or 'Login' on the current view.")

        print("Test complete. Closing browser.")
        await browser.close()

if __name__ == '__main__':
    # Run the async function
    asyncio.run(run_e2e_test())
