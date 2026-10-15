#!/usr/bin/env python3
"""Busca noticias del mundo por RSS, las resume en español con Gemini (nivel gratis)
y regenera la página estática en docs/index.html.

Uso:
    GEMINI_API_KEY=xxxx python fetch_news.py
    python fetch_news.py --dry-run      # sin IA, solo titulares (para probar)
"""
import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests

ROOT = Path(__file__).parent
DOCS = ROOT / "docs"
DATA = DOCS / "data" / "noticias.json"

# Fuentes RSS de noticias internacionales. Puedes añadir o quitar las que quieras.
FEEDS = [
    ("BBC Mundo", "https://feeds.bbci.co.uk/mundo/rss.xml"),
    ("DW Español", "https://rss.dw.com/rdf/rss-sp-all"),
    ("France 24", "https://www.france24.com/es/rss"),
    ("El País - Internacional", "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/internacional/portada"),
    ("Reuters World (vía Google News)", "https://news.google.com/rss/search?q=world+when:1d&hl=es-419&gl=DO&ceid=DO:es-419"),
]

MAX_PER_FEED = 4          # noticias nuevas por fuente en cada ejecución
MAX_NEW_PER_RUN = 12      # tope total por ejecución (cuida el límite gratis de la IA)
MAX_STORED = 150          # noticias que se conservan en la página
API = "https://generativelanguage.googleapis.com/v1beta"
SITE_NAME = os.environ.get("SITE_NAME", "Mundo al Día")


def clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def uid(link: str) -> str:
    return hashlib.sha1(link.encode()).hexdigest()[:12]


def load_existing() -> list:
    if DATA.exists():
        try:
            return json.loads(DATA.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return []


def collect_candidates(seen: set) -> list:
    found = []
    for source, url in FEEDS:
        try:
            parsed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0 NoticiasBot"})
        except Exception as exc:  # una fuente caída no debe frenar todo
            print(f"[aviso] {source}: {exc}", file=sys.stderr)
            continue
        count = 0
        for entry in parsed.entries:
            link = entry.get("link")
            if not link or uid(link) in seen:
                continue
            found.append({
                "id": uid(link),
                "fuente": source,
                "enlace": link,
                "titulo_original": clean(entry.get("title", "")),
                "extracto": clean(entry.get("summary", ""))[:500],
            })
            count += 1
            if count >= MAX_PER_FEED:
                break
    # Intercala fuentes para que haya variedad
    found.sort(key=lambda x: x["id"])
    return found[:MAX_NEW_PER_RUN]


def summarize_with(items: list, api_key: str, model: str) -> list:
    """Pide a Gemini un titular y resumen propios (con sus palabras) por noticia."""
    prompt = (
        "Eres editor de un sitio de noticias internacionales en español. Para cada noticia "
        "escribe un titular nuevo y un resumen de 2 a 3 frases con TUS PROPIAS palabras, "
        "usando únicamente la información dada (no inventes datos, cifras ni citas). "
        "Asigna una categoría: Política, Economía, Sociedad, Ciencia y Tecnología, "
        "Deportes, Cultura, Conflictos o Medio Ambiente. "
        "Responde SOLO con un arreglo JSON, en el mismo orden, con objetos "
        '{"titulo": "...", "resumen": "...", "categoria": "..."}.\n\n'
        + json.dumps(
            [{"titulo": i["titulo_original"], "extracto": i["extracto"], "fuente": i["fuente"]} for i in items],
            ensure_ascii=False,
        )
    )
    url = f"{API}/models/{model}:generateContent"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3},
    }
    for attempt in range(3):
        resp = requests.post(url, params={"key": api_key}, json=body, timeout=90)
        if resp.status_code == 429:  # límite del nivel gratis: esperar y reintentar
            time.sleep(20 * (attempt + 1))
            continue
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(text)
        if isinstance(result, list) and len(result) == len(items):
            return result
        raise ValueError("La IA devolvió un formato inesperado")
    raise RuntimeError("Límite de la IA alcanzado; se intentará en la próxima ejecución")


def pick_models(api_key: str) -> list:
    """Pregunta a Google qué modelos 'flash' existen hoy y devuelve los mejores primero."""
    forced = os.environ.get("GEMINI_MODEL")
    if forced:
        return [forced]
    resp = requests.get(f"{API}/models", params={"key": api_key, "pageSize": 200}, timeout=30)
    resp.raise_for_status()
    skip = ("image", "tts", "live", "audio", "embedding", "robotics", "computer", "native", "thinking")
    stable, preview = [], []
    for m in resp.json().get("models", []):
        if "generateContent" not in m.get("supportedGenerationMethods", []):
            continue
        name = m["name"].split("/")[-1]
        if "flash" not in name or any(s in name for s in skip):
            continue
        (preview if ("preview" in name or "exp" in name) else stable).append(name)

    def rank(n: str):
        found = re.search(r"(\d+(?:\.\d+)?)", n)
        version = float(found.group(1)) if found else 0.0
        return ("lite" in n, -version)

    models = sorted(stable, key=rank) + sorted(preview, key=rank)
    if not models:
        raise RuntimeError("No se encontró ningún modelo Gemini disponible para esta clave")
    return models[:4]


def summarize(items: list, api_key: str) -> list:
    last_error = None
    for model in pick_models(api_key):
        try:
            print(f"Usando modelo: {model}")
            return summarize_with(items, api_key, model)
        except requests.HTTPError as exc:
            last_error = exc
            if exc.response is not None and exc.response.status_code in (400, 404):
                continue  # ese modelo no sirve; probar el siguiente
            raise
    raise RuntimeError(f"Ningún modelo de Gemini funcionó: {last_error}")


def render(items: list) -> str:
    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    cards = []
    for n in items:
        cards.append(
            f"""<article class="card">
  <span class="tag">{html.escape(n.get('categoria', 'General'))}</span>
  <h2><a href="{html.escape(n['enlace'])}" target="_blank" rel="noopener nofollow">{html.escape(n['titulo'])}</a></h2>
  <p>{html.escape(n['resumen'])}</p>
  <footer>Fuente: <a href="{html.escape(n['enlace'])}" target="_blank" rel="noopener nofollow">{html.escape(n['fuente'])}</a>
  · {html.escape(n['fecha'][:10])}</footer>
</article>"""
        )
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(SITE_NAME)} · Noticias del mundo</title>
<meta name="description" content="Resúmenes automáticos de las noticias más importantes del mundo, con enlace a la fuente original.">
<style>
:root{{--bg:#f6f7f9;--fg:#15181d;--card:#fff;--muted:#5b6472;--accent:#0a58ca;--line:#e3e6eb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0f1115;--fg:#e8eaee;--card:#171a21;--muted:#9aa3b2;--accent:#6ea8fe;--line:#262b36}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
header{{padding:28px 16px 12px;max-width:820px;margin:0 auto}}
h1{{margin:0;font-size:1.8rem}}
header p{{margin:4px 0 0;color:var(--muted);font-size:.9rem}}
main{{max-width:820px;margin:0 auto;padding:8px 16px 40px;display:grid;gap:14px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}}
.card h2{{font-size:1.15rem;margin:8px 0 6px;line-height:1.3}}
.card a{{color:inherit;text-decoration:none}}
.card h2 a:hover{{color:var(--accent)}}
.card p{{margin:0 0 10px}}
.tag{{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:var(--accent);font-weight:600}}
footer{{color:var(--muted);font-size:.82rem}}
footer a{{color:var(--accent)}}
.site-foot{{text-align:center;color:var(--muted);font-size:.8rem;padding:0 16px 32px}}
</style>
</head>
<body>
<header>
<h1>{html.escape(SITE_NAME)}</h1>
<p>Noticias del mundo resumidas automáticamente con IA · Actualizado: {now}</p>
</header>
<main>
{chr(10).join(cards)}
</main>
<p class="site-foot">Resúmenes generados por IA a partir de fuentes públicas. Pueden contener errores: consulta siempre la fuente original.</p>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="No usa IA; publica solo titulares")
    args = parser.parse_args()

    existing = load_existing()
    seen = {n["id"] for n in existing}
    candidates = collect_candidates(seen)
    print(f"Noticias nuevas encontradas: {len(candidates)}")

    new_items = []
    if candidates:
        if args.dry_run:
            summaries = [{"titulo": c["titulo_original"], "resumen": c["extracto"][:200], "categoria": "General"} for c in candidates]
        else:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print("Falta la variable GEMINI_API_KEY", file=sys.stderr)
                return 1
            summaries = summarize(candidates, api_key)
        today = datetime.now(timezone.utc).isoformat()
        for c, s in zip(candidates, summaries):
            new_items.append({
                "id": c["id"],
                "fuente": c["fuente"],
                "enlace": c["enlace"],
                "titulo": clean(s.get("titulo", c["titulo_original"])),
                "resumen": clean(s.get("resumen", "")),
                "categoria": clean(s.get("categoria", "General")),
                "fecha": today,
            })

    all_items = (new_items + existing)[:MAX_STORED]
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(all_items, ensure_ascii=False, indent=1), encoding="utf-8")
    (DOCS / "index.html").write_text(render(all_items), encoding="utf-8")
    print(f"Página actualizada con {len(all_items)} noticias")
    return 0


if __name__ == "__main__":
    sys.exit(main())
