import os
import sys
import time
import subprocess
import socket

import numpy as np
import pyautogui as pag

try:
    import pygetwindow as gw
except ImportError:
    gw = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from action.base import BaseController
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

DOWNLOAD_DIR = os.path.join(os.path.expanduser("~"), "Downloads")
DEBUGGING_PORT = 9222  # Chrome remote-debugging port
DEFAULT_CHROME_PROFILE = os.environ.get("CHROME_PROFILE_DIRECTORY", "Default")
DEFAULT_AUTOMATION_USER_DATA_DIR = os.environ.get(
    "CHROME_AUTOMATION_USER_DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory_store", "chrome_automation_profile"),
)


def _is_port_open(port: int) -> bool:
    """Check if a local TCP port is already listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _find_chrome_path() -> str:
    """Return the path to the Chrome executable on Windows."""
    candidates = [
        os.path.expandvars(
            r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"
        ),
        os.path.expandvars(
            r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
        ),
        os.path.expandvars(
            r"%LocalAppData%\Google\Chrome\Application\chrome.exe"
        ),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return "chrome"  # hope it's on PATH


class BrowserController(BaseController):
    def __init__(self):
        self.driver = None
        self._profile_reader = None

    def is_available(self) -> bool:
        return self.is_running

    @property
    def is_running(self):
        if self.driver is None:
            return False
        try:
            # Quick health check — if the browser crashed this will throw
            _ = self.driver.title
            return True
        except Exception:
            self.driver = None
            return False

    # ── Launch / attach ──────────────────────────────────────────────

    def start_browser(self):
        """Attach to an already-running Chrome, or launch one with the
        user's real profile so it looks like a normal browsing session
        (bypasses most bot / captcha detection)."""

        # 1. If Chrome is not already running with debugging, start it
        if not _is_port_open(DEBUGGING_PORT):
            self._launch_debug_chrome(DEFAULT_AUTOMATION_USER_DATA_DIR)

            # If the port still did not open, retry once with a fresh automation profile.
            if not _is_port_open(DEBUGGING_PORT):
                fallback_user_data = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "memory_store",
                    "chrome_automation_profile_retry",
                )
                self._launch_debug_chrome(fallback_user_data)

            if not _is_port_open(DEBUGGING_PORT):
                raise RuntimeError("Chrome remote debugging port 9222 did not open")

        # 2. Connect Selenium to the running Chrome instance
        opts = Options()
        opts.debugger_address = f"127.0.0.1:{DEBUGGING_PORT}"

        try:
            self.driver = webdriver.Chrome(options=opts)
        except Exception:
            self.driver = None
            raise
        self._dismiss_profile_picker()
        print("[Browser] Attached to Chrome (real profile, anti-captcha).")

    def _launch_debug_chrome(self, user_data_dir: str) -> None:
        chrome_path = _find_chrome_path()
        os.makedirs(user_data_dir, exist_ok=True)
        profile_dir = os.environ.get("CHROME_PROFILE_DIRECTORY", DEFAULT_CHROME_PROFILE)
        cmd = [
            chrome_path,
            f"--remote-debugging-port={DEBUGGING_PORT}",
            f"--user-data-dir={user_data_dir}",
            f"--profile-directory={profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--start-maximized",
        ]
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Give Chrome a moment to spin up the debug listener.
        for _ in range(20):
            if _is_port_open(DEBUGGING_PORT):
                break
            time.sleep(0.5)

    def _get_profile_reader(self):
        if self._profile_reader is None:
            import easyocr

            self._profile_reader = easyocr.Reader(["en"], gpu=False)
        return self._profile_reader

    def _profile_picker_window_visible(self) -> bool:
        if gw is None:
            return False
        try:
            titles = [w.title.lower() for w in gw.getAllWindows() if w.title]
        except Exception:
            return False
        markers = (
            "choose your profile",
            "choose a profile",
            "profile picker",
            "who's using chrome",
            "who is using chrome",
        )
        return any(any(marker in title for marker in markers) for title in titles)

    def _dismiss_profile_picker(self) -> bool:
        """Try to close Chrome's native profile chooser by clicking a profile tile."""
        if not self._profile_picker_window_visible():
            return False

        target_profile = os.environ.get("CHROME_PROFILE_DIRECTORY", DEFAULT_CHROME_PROFILE).strip().lower()
        blocked_text = {
            "add", "guest", "sign in", "continue as guest", "other", "manage people",
        }

        try:
            reader = self._get_profile_reader()
        except Exception:
            return False

        for _ in range(3):
            try:
                screenshot = pag.screenshot()
                results = reader.readtext(np.array(screenshot))
            except Exception:
                return False

            fallback = None
            for box, text, confidence in results:
                label = text.strip()
                normalized = label.lower()
                if not label or confidence < 0.35:
                    continue
                if target_profile and target_profile in normalized:
                    x = int(sum(point[0] for point in box) / 4)
                    y = int(sum(point[1] for point in box) / 4)
                    pag.click(x, y)
                    time.sleep(2)
                    return True
                if fallback is None and normalized not in blocked_text and len(label) <= 40:
                    fallback = (box, label)

            if fallback is not None:
                box, _ = fallback
                x = int(sum(point[0] for point in box) / 4)
                y = int(sum(point[1] for point in box) / 4)
                pag.click(x, y)
                time.sleep(2)
                return True

            time.sleep(0.5)

        return False

    # ── Navigation ───────────────────────────────────────────────────

    def navigate(self, url: str):
        if not url.startswith("http"):
            url = "https://" + url
        self.driver.get(url)
        WebDriverWait(self.driver, 15).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )

    def search(self, query: str):
        from urllib.parse import quote_plus

        self.navigate(f"https://www.google.com/search?q={quote_plus(query)}")
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "search"))
            )
        except Exception as e:
            print(f"[Browser] Search failed: {e}")

    # ── Element extraction (DOM-based, same data shape as before) ────

    def get_page_elements(self, max_elements: int = 80) -> list:
        js = """
        const elements = [];
        const seen = new Set();
        const selectors = [
            'a[href]', 'button', 'input', 'textarea', 'select',
            '[role="button"]', '[role="link"]', '[onclick]',
            'h1', 'h2', 'h3', 'summary', '[download]'
        ];
        for (const sel of selectors) {
            for (const el of document.querySelectorAll(sel)) {
                if (seen.has(el)) continue;
                seen.add(el);
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) continue;
                if (rect.top > window.innerHeight + 100) continue;
                const text = (el.innerText || el.value || el.getAttribute('aria-label')
                              || el.getAttribute('title') || el.alt || '').trim().slice(0, 120);
                if (!text) continue;
                let tag = el.tagName.toLowerCase();
                let type = 'link';
                if (tag === 'button' || el.getAttribute('role') === 'button') type = 'button';
                else if (tag === 'input' || tag === 'textarea') type = 'input';
                else if (tag === 'select') type = 'select';
                else if (['h1','h2','h3'].includes(tag)) type = 'heading';
                const href = el.getAttribute('href') || '';
                elements.push({
                    tag, type, text, href,
                    center_x: Math.round(rect.x + rect.width / 2),
                    center_y: Math.round(rect.y + rect.height / 2),
                });
                if (elements.length >= """ + str(max_elements) + """) break;
            }
            if (elements.length >= """ + str(max_elements) + """) break;
        }
        return elements;
        """
        try:
            return self.driver.execute_script(js)
        except Exception as e:
            print(f"[Browser] get_page_elements failed: {e}")
            return []

    def get_page_text(self, max_chars: int = 1500) -> str:
        js = """
        const candidates = document.querySelectorAll(
            'article, main, [role="main"], .content, .post-content, .entry-content, .recipe-content'
        );
        let root = candidates.length > 0 ? candidates[0] : document.body;
        let text = root.innerText || '';
        text = text.replace(/\\n{3,}/g, '\\n\\n').trim();
        return text.slice(0, """ + str(max_chars) + """);
        """
        try:
            return self.driver.execute_script(js)
        except Exception as e:
            print(f"[Browser] get_page_text failed: {e}")
            return ""

    def click_element(self, index: int, elements: list) -> str:
        if index < 0 or index >= len(elements):
            return f"Invalid index {index}"
        el = elements[index]
        try:
            x, y = el["center_x"], el["center_y"]
            self.driver.execute_script(
                "document.elementFromPoint(arguments[0], arguments[1])?.click();",
                x, y,
            )
            time.sleep(1)
            return f'Clicked "{el["text"][:60]}" OK'
        except Exception as e:
            return f"Click failed: {e}"

    def type_text(self, text: str):
        actions = ActionChains(self.driver)
        actions.send_keys(text).perform()

    def press_key(self, key: str):
        key_map = {
            "Enter": Keys.RETURN,
            "Tab": Keys.TAB,
            "Escape": Keys.ESCAPE,
            "Backspace": Keys.BACKSPACE,
            "ArrowDown": Keys.ARROW_DOWN,
            "ArrowUp": Keys.ARROW_UP,
        }
        actions = ActionChains(self.driver)
        actions.send_keys(key_map.get(key, key)).perform()

    def scroll_page(self, direction: str = "down", amount: int = 500):
        delta = amount if direction == "down" else -amount
        self.driver.execute_script(f"window.scrollBy(0, {delta});")
        time.sleep(0.4)

    def go_back(self):
        self.driver.back()
        time.sleep(1)

    def get_current_url(self) -> str:
        return self.driver.current_url

    def get_page_title(self) -> str:
        return self.driver.title

    # ── Download support ─────────────────────────────────────────────

    def download_file(self, index: int, elements: list) -> str:
        if index < 0 or index >= len(elements):
            return f"Invalid index {index}"
        el = elements[index]
        try:
            # Just click — Chrome downloads natively with the user profile
            result = self.click_element(index, elements)
            time.sleep(3)
            return f"Download triggered for \"{el['text'][:60]}\""
        except Exception as e:
            return f"Download failed: {e}"

    def take_screenshot(self) -> bytes:
        return self.driver.get_screenshot_as_png()

    def close_browser(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
        self.driver = None