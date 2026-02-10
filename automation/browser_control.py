from playwright.sync_api import sync_playwright
import pyautogui as pag

class BrowserController:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None
    
    def start_browser(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=False,
            args=[
                "--no-default-browser-check",
                "--disable-default-apps",
                "--disable-blink-features=AutomationControlled",
                "--disable-web-resources",
                "--disable-extensions",
            ]
        )
        self.page = self.browser.new_page(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        # Stealth mode: hide automation indicators
        self.page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => false});")
        self.page.add_init_script("Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});")
        self.page.add_init_script("Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});")
        self.page.add_init_script("window.chrome = {runtime: {}};")

    
    def navigate(self, url: str):
        if not url.startswith("http"):
            url = "http://" + url
        self.page.goto(url)
    
    def search(self, query: str):
        self.page.goto("https://www.google.com", wait_until="domcontentloaded")
        self.page.wait_for_timeout(3000)
        
        # Check if reCAPTCHA is present and wait for it to be solved
        try:
            self.page.wait_for_selector("iframe[title='recaptcha']", timeout=2000)
            print("reCAPTCHA detected - waiting 10 seconds for manual solving...")
            self.page.wait_for_timeout(10000)
        except:
            pass
        
        # Try multiple selectors for the search box
        try:
            self.page.wait_for_selector("textarea[name='q']", timeout=5000)
            self.page.fill("textarea[name='q']", query)
        except:
            try:
                self.page.wait_for_selector("input[aria-label*='Search']", timeout=5000)
                self.page.fill("input[aria-label*='Search']", query)
            except:
                self.page.fill("input[name='q']", query)
        
        self.page.keyboard.press("Enter")
        self.page.wait_for_timeout(2000)
    
    def click(self):
        self.page.wait_for_selector("h3")
        self.page.click("h3")
    
    def close_browser(self):
        self.browser.close()
        self.playwright.stop()