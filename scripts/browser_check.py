"""Optional browser check of rendered equations, images, and explorer behavior."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import sys
import threading
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/".vendor").exists():
    sys.path.insert(0,str(ROOT/".vendor"))
from playwright.sync_api import sync_playwright

class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self,*args):
        pass

def main():
    server=ThreadingHTTPServer(("127.0.0.1",0),partial(QuietHandler,directory=str(ROOT/"site")))
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    base=f"http://127.0.0.1:{server.server_port}"
    candidates=[os.environ.get("BROWSER_EXECUTABLE"),shutil.which("google-chrome"),shutil.which("chromium"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]
    executable=next((p for p in candidates if p and Path(p).is_file()),None)
    out=ROOT/"artifacts/browser"
    out.mkdir(parents=True,exist_ok=True)
    report=dict(pages=[],math_expressions=0,errors=[],explorer={},design={})
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path=executable,headless=True)
            page=browser.new_page(viewport={"width":1440,"height":1000},device_scale_factor=1)
            page.on("pageerror",lambda err:report["errors"].append(str(err)))
            for source in sorted((ROOT/"docs").glob("*.md")):
                route="/" if source.stem=="index" else "/"+source.stem+"/"
                page.goto(base+route,wait_until="networkidle")
                assert page.locator("body").get_attribute("data-md-color-scheme")=="slate"
                assert page.evaluate("getComputedStyle(document.body).backgroundColor")=="rgb(23, 25, 29)"
                assert page.locator('[data-md-component="palette"]').count()==0
                assert page.locator(".book-wordmark").inner_text()=="Positional Encoding in LLMs"
                assert "Positional Encoding in LLMs" in page.title()
                count=page.locator(".arithmatex").count()
                if count:
                    page.wait_for_function("window.MathJax && MathJax.startup && MathJax.startup.document && document.querySelectorAll('mjx-container').length >= document.querySelectorAll('.arithmatex').length",timeout=30000)
                assert page.locator("[data-mjx-error],mjx-merror").count()==0,f"invalid equation: {route}"
                assert page.locator("img").evaluate_all("(imgs)=>imgs.every(i=>i.complete && i.naturalWidth>0)"),f"broken image: {route}"
                report["math_expressions"]+=count
                report["pages"].append(dict(route=route,equations=count))
                assert page.locator("img").evaluate_all("(imgs)=>imgs.every(i=>new URL(i.src).pathname.endsWith('.svg'))")
                if source.stem in ["index","04-decay","09-results"]:
                    page.screenshot(path=str(out/(source.stem+".png")),full_page=False)
            page.goto(base+"/",wait_until="networkidle")
            for preference in ["light","dark"]:
                page.emulate_media(color_scheme=preference)
                page.reload(wait_until="networkidle")
                assert page.evaluate("getComputedStyle(document.body).backgroundColor")=="rgb(23, 25, 29)"
                assert page.locator('[data-md-component="palette"]').count()==0
            page.locator(".book-sections summary").click()
            assert page.locator(".book-sections nav a").count()==16
            page.locator(".book-sections summary").click()
            page.locator(".md-search__input").fill("decay")
            page.wait_for_selector(".md-search-result__item")
            page.keyboard.press("Escape")
            page.set_viewport_size({"width":375,"height":812})
            page.reload(wait_until="networkidle")
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"),"book overflows mobile"
            page.screenshot(path=str(out/"book-mobile.png"))
            page.locator(".book-sections summary").click()
            assert page.locator(".book-sections nav").is_visible()
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"),"section index overflows mobile"
            page.locator(".book-sections summary").click()
            page.set_viewport_size({"width":1440,"height":1000})
            report["design"]=dict(permanent_dark=True,no_palette_control=True,system_preference_independent=True,
                                  svg_figures=True,section_menu=True,search=True,mobile=True)
            page.goto(base+"/explorer.html",wait_until="networkidle")
            assert page.evaluate("getComputedStyle(document.body).backgroundColor")=="rgb(23, 25, 29)"
            assert page.locator("h1").inner_text()=="Position-dependent rotations and phase similarity"
            original=page.locator("#similarity").inner_text()
            delta=page.locator("#delta").inner_text()
            page.locator("#shift").click()
            assert page.locator("#similarity").inner_text()==original
            assert page.locator("#delta").inner_text()==delta
            page.locator("#scale").select_option("4")
            assert page.locator("#similarity").inner_text()!=original
            page.locator("#query").fill("0")
            page.locator("#key").fill("0")
            assert page.locator("#similarity").inner_text()=="1.000000"
            assert page.locator("#distance").inner_text()=="0.000000"
            page.locator("#query").focus()
            page.keyboard.press("ArrowRight")
            assert page.locator("#queryValue").inner_text()=="1"
            page.locator("#reset").click()
            page.screenshot(path=str(out/"explorer-desktop.png"),full_page=True)
            page.set_viewport_size({"width":375,"height":812})
            page.screenshot(path=str(out/"explorer-mobile.png"),full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"),"explorer overflows mobile"
            report["explorer"]=dict(common_shift=True,interpolation=True,zero_offset=True,
                                    keyboard=True,mobile_no_overflow=True)
            assert not report["errors"],report["errors"]
            browser.close()
    finally:
        server.shutdown()
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Browser verified {len(report['pages'])} dark-theme pages and {report['math_expressions']} equations; fixed theme, search, navigation, SVG figures, explorer, and mobile checks pass.")

if __name__=="__main__":
    main()
