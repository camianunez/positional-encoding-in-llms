"""Check the book's links, source identity, figure provenance, and optional built HTML."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit
ROOT=Path(__file__).resolve().parents[1]

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ("href","src") and value:
                self.targets.append(value)

def check_target(source,target,base=None):
    target=unquote(target.split("#")[0].split("?")[0])
    if not target or urlsplit(target).scheme or target.startswith("//"):
        return
    if target.startswith("/"):
        if base is None:
            raise AssertionError(f"unportable absolute local link in {source}: {target}")
        target=target.removeprefix("/positional-encoding-in-llms/")
        dest=base/target.lstrip("/")
    else:
        dest=source.parent/target
    if dest.is_dir():
        dest=dest/"index.html"
    assert dest.exists(),f"missing link in {source.relative_to(ROOT)}: {target}"

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site",action="store_true")
    args=parser.parse_args()
    manifest=json.loads((ROOT/"experiments/paper-manifest.json").read_text())
    assert hashlib.sha256((ROOT/manifest["file"]).read_bytes()).hexdigest()==manifest["sha256"]
    assert manifest["pages"]==8
    docs=sorted((ROOT/"docs").glob("*.md"))
    assert len(docs)==15,f"expected 15 book pages, got {len(docs)}"
    words=0
    for p in [ROOT/"README.md",*docs]:
        content=p.read_text(encoding="utf-8")
        assert not re.search(r"\b(TODO|TBD|FIXME)\b",content),f"unfinished text: {p}"
        assert r"\(" not in content,f"use GitHub-compatible inline dollar math in {p}"
        assert content.count("$$")%2==0,f"unpaired display math: {p}"
        words+=len(content.split())
        for target in re.findall(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)",content):
            check_target(p,target)
    config=(ROOT/"mkdocs.yml").read_text(encoding="utf-8")
    assert "site_name: Positional Encoding in LLMs" in config
    assert "scheme: slate" in config and "toggle:" not in config
    assert (ROOT/"overrides/partials/header.html").is_file()
    assert (ROOT/"overrides/partials/footer.html").is_file()
    for p in docs:
        assert p.name in config,f"missing navigation page: {p.name}"
    graph_manifest=json.loads((ROOT/"experiments/graph-manifest.json").read_text())
    assert len(graph_manifest)==10
    for item in graph_manifest:
        for suffix in ["png","svg"]:
            path=ROOT/"docs/assets"/(item["figure"]+"."+suffix)
            assert path.is_file() and path.stat().st_size>1000,f"missing or empty graph: {path}"
        svg=(ROOT/"docs/assets"/(item["figure"]+".svg")).read_text(encoding="utf-8")
        assert "<image" not in svg,"raster content in vector-primary figure"
        assert (ROOT/item["script"]).is_file()
        assert item["preferred_format"]=="svg"
        assert (ROOT/item["style"]).is_file()
        for data in item["data"]:
            assert (ROOT/data).exists()
    figure_env=json.loads((ROOT/"experiments/diagnostics/figure-environment.json").read_text())
    for name,script in [("report","report.py"),("diagnostics","diagnostics.py"),("style","figure_style.py")]:
        assert hashlib.sha256((ROOT/"scripts"/script).read_bytes()).hexdigest()==figure_env[name+"_source_sha256"]
    explorer=ROOT/"docs/explorer.html"
    html=explorer.read_text(encoding="utf-8")
    assert "canvas" in html and "aria-live" in html and "<script>" in html
    assert "<script src" not in html,"explorer must be self-contained"
    if args.site:
        site=ROOT/"site"
        for p in site.rglob("*.html"):
            links=Links()
            links.feed(p.read_text(encoding="utf-8"))
            for target in links.targets:
                check_target(p,target,site)
        for p in docs:
            destination=site/"index.html" if p.stem=="index" else site/p.stem/"index.html"
            assert destination.is_file()
    print(f"Verified {len(docs)} pages, {len(graph_manifest)} figures in both formats, PDF identity, navigation and local links; approximately {words:,} source words.")
    if args.site:
        print("All built HTML resource links also resolve.")

if __name__=="__main__":
    main()
