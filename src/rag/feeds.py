"""Coleta de notícias via feeds RSS públicos (sem raspagem de HTML).

Princípios: usar apenas o conteúdo que o próprio feed publica, respeitar o robots.txt,
identificar-se com um User-Agent claro e limitar a taxa de requisições.
"""

import calendar
import hashlib
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import feedparser
import requests
from bs4 import BeautifulSoup

USER_AGENT = "Taizen.IA-research/0.1 (+https://github.com/gtwave/Taizen.IA)"
REQUEST_DELAY_S = 1.0
TIMEOUT_S = 20


@dataclass(frozen=True)
class Feed:
    name: str
    url: str
    pages: int  # quantas páginas do feed (?paged=N) coletar


FEEDS = [
    Feed("infomoney", "https://www.infomoney.com.br/feed/", 8),
    Feed("moneytimes", "https://www.moneytimes.com.br/feed/", 8),
    Feed("seudinheiro", "https://www.seudinheiro.com/feed/", 8),
    Feed("exame_invest", "https://exame.com/invest/feed/", 3),
    Feed("einvestidor", "https://einvestidor.estadao.com.br/feed/", 1),
]

_robots_cache: dict[str, RobotFileParser | None] = {}


def robots_allows(url: str, session: requests.Session) -> bool:
    """True se o robots.txt do host permite buscar a URL com o nosso User-Agent."""
    parts = urlparse(url)
    base = f"{parts.scheme}://{parts.netloc}"
    if base not in _robots_cache:
        rp = RobotFileParser()
        try:
            resp = session.get(f"{base}/robots.txt", timeout=TIMEOUT_S)
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
                _robots_cache[base] = rp
            else:
                _robots_cache[base] = None  # sem robots.txt: sem restrições declaradas
        except requests.RequestException:
            _robots_cache[base] = None
    rp = _robots_cache[base]
    return True if rp is None else rp.can_fetch(USER_AGENT, url)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/xml;q=0.9"})
    return s


def page_url(feed: Feed, page: int) -> str:
    return feed.url if page == 1 else f"{feed.url}?paged={page}"


_BLOCK_TAGS = ["p", "div", "li", "ul", "ol", "br", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table"]


def html_to_text(html: str) -> str:
    """Converte o HTML do feed em texto limpo: quebra linha só em blocos, não em links/negrito."""
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "figure", "iframe"]):
        tag.decompose()
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    text = soup.get_text()
    lines = [re.sub(r"[ \t\u00a0]+", " ", ln).strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", text))


def article_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def parse_entries(xml: bytes | str, source: str) -> list[dict]:
    """Normaliza as entradas de um feed RSS para o formato interno de artigo."""
    parsed = feedparser.parse(xml)
    out = []
    for e in parsed.entries:
        url = e.get("link")
        if not url:
            continue
        body = ""
        if e.get("content"):
            body = e.content[0].get("value", "")
        if not body:
            body = e.get("summary", "")
        text = html_to_text(body)
        ts = None
        if e.get("published_parsed"):
            ts = calendar.timegm(e.published_parsed)
        published = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None
        out.append({
            "id": article_id(url),
            "source": source,
            "title": html_to_text(e.get("title", "")).replace("\n", " ").strip(),
            "url": url,
            "published": published,
            "published_ts": ts,
            "text": text,
        })
    return out


def fetch_feed(feed: Feed, session: requests.Session, log=print) -> list[dict]:
    """Baixa as páginas de um feed (com robots.txt e intervalo entre requisições)."""
    articles: list[dict] = []
    for page in range(1, feed.pages + 1):
        url = page_url(feed, page)
        if not robots_allows(url, session):
            log(f"  [{feed.name}] robots.txt não permite {url} — ignorado")
            break
        try:
            resp = session.get(url, timeout=TIMEOUT_S)
            if resp.status_code != 200:
                log(f"  [{feed.name}] página {page}: HTTP {resp.status_code} — encerrando este feed")
                break
        except requests.RequestException as exc:
            log(f"  [{feed.name}] página {page}: erro de rede ({exc.__class__.__name__}) — encerrando este feed")
            break
        entries = parse_entries(resp.content, feed.name)
        articles.extend(entries)
        if not entries:
            break
        time.sleep(REQUEST_DELAY_S)
    return articles
