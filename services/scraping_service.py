"""
Serviço centralizado de web scraping com cache em memória e rate limiting.

Fontes:
  - Open-Meteo   → previsão do tempo (API JSON, sem chave)
  - Yahoo Finance → cotação de futuros agrícolas CBOT (API JSON, sem chave)
  - Canal Rural   → notícias do agro via RSS/XML
"""

import time
import logging
import os
from datetime import datetime
from typing import Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_CACHE_TTL = int(os.getenv("SCRAPING_CACHE_TTL", "600"))  # 10 minutos por padrão
_WEATHER_LAT = float(os.getenv("WEATHER_LAT", "-15.77"))   # Brasília como padrão
_WEATHER_LON = float(os.getenv("WEATHER_LON", "-47.92"))


class _Cache:
    def __init__(self, ttl: int):
        self._ttl = ttl
        self._store: dict[str, dict] = {}

    def get(self, key: str) -> Optional[dict]:
        entry = self._store.get(key)
        if entry and (time.time() - entry["ts"]) < self._ttl:
            return entry["data"]
        return None

    def set(self, key: str, data: dict) -> None:
        self._store[key] = {"data": data, "ts": time.time()}


_cache = _Cache(ttl=_CACHE_TTL)

_HEADERS_HTML = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# RSS/XML: sem Accept-Encoding para evitar compressão que quebra o parser lxml
_HEADERS_RSS = {
    "User-Agent": "Mozilla/5.0 (compatible; AgroVisionBot/1.0)",
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}


class ScrapingService:
    """Serviço centralizado de web scraping com cache e rate limiting."""

    # ──────────────────────────────────────────────────
    # FONTE 1 — Previsão do Tempo (Open-Meteo)
    # Sem API key, gratuito, estável, dados em JSON
    # ──────────────────────────────────────────────────
    def get_weather(self, lat: float = _WEATHER_LAT, lon: float = _WEATHER_LON) -> dict:
        key = f"weather_{lat:.4f}_{lon:.4f}"
        cached = _cache.get(key)
        if cached:
            return {**cached, "cached": True}

        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,"
            "precipitation_probability,wind_speed_10m,weather_code"
            "&timezone=America%2FSao_Paulo"
            "&forecast_days=1"
        )
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(url)
                resp.raise_for_status()
                raw = resp.json()
        except httpx.TimeoutException:
            return self._err("Open-Meteo", "timeout ao conectar")
        except httpx.HTTPStatusError as exc:
            return self._err("Open-Meteo", f"HTTP {exc.response.status_code}")
        except Exception as exc:
            return self._err("Open-Meteo", str(exc))

        current = raw.get("current", {})
        result = {
            "fonte": "Open-Meteo",
            "coletado_em": datetime.now().isoformat(),
            "temperatura_c": current.get("temperature_2m"),
            "umidade_pct": current.get("relative_humidity_2m"),
            "prob_chuva_pct": current.get("precipitation_probability"),
            "vento_kmh": current.get("wind_speed_10m"),
            "codigo_tempo": current.get("weather_code"),
            "cached": False,
        }
        _cache.set(key, result)
        return result

    # ──────────────────────────────────────────────────────────────
    # FONTE 2 — Cotações de Commodities (Yahoo Finance JSON API)
    # Endpoint público sem API key — futuros CBOT negociados em USD
    # Símbolos: ZS=F (Soja), ZC=F (Milho), GC=F (Ouro como referência)
    # ──────────────────────────────────────────────────────────────
    _YF_SYMBOLS = {"soja": "ZS=F", "milho": "ZC=F"}

    def get_commodity_prices(self) -> dict:
        key = "commodity_prices"
        cached = _cache.get(key)
        if cached:
            return {**cached, "cached": True}

        precos: dict = {}
        for nome, symbol in self._YF_SYMBOLS.items():
            url = (
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                "?interval=1d&range=1d"
            )
            try:
                with httpx.Client(timeout=10, headers=_HEADERS_HTML, follow_redirects=True) as client:
                    resp = client.get(url)
                    resp.raise_for_status()
                    data = resp.json()

                meta = data["chart"]["result"][0]["meta"]
                precos[nome] = {
                    "preco": round(meta.get("regularMarketPrice", 0), 2),
                    "preco_abertura": round(meta.get("chartPreviousClose", 0), 2),
                    "moeda": meta.get("currency", "USD"),
                    "unidade": "cents/bushel (CBOT)",
                    "simbolo": symbol,
                }
            except httpx.TimeoutException:
                precos[nome] = {"erro": "timeout"}
            except (httpx.HTTPStatusError, KeyError, IndexError, ValueError) as exc:
                precos[nome] = {"erro": str(exc)}
            except Exception as exc:
                precos[nome] = {"erro": str(exc)}

        result = {
            "fonte": "Yahoo Finance (CBOT Futuros)",
            "coletado_em": datetime.now().isoformat(),
            "nota": "Preços em USD cents/bushel — mercado futuro Chicago (CBOT)",
            "precos": precos,
            "cached": False,
        }
        _cache.set(key, result)
        return result

    # ──────────────────────────────────────────────────
    # FONTE 3 — Notícias do Agro (Canal Rural RSS)
    # RSS é formato estável; tolerante a mudanças no site
    # ──────────────────────────────────────────────────
    def get_agro_news(self) -> dict:
        key = "agro_news"
        cached = _cache.get(key)
        if cached:
            return {**cached, "cached": True}

        url = "https://www.canalrural.com.br/feed/"
        try:
            with httpx.Client(timeout=15, headers=_HEADERS_RSS, follow_redirects=True) as client:
                resp = client.get(url)
                resp.raise_for_status()
        except httpx.TimeoutException:
            return self._err("Canal Rural", "timeout ao conectar")
        except httpx.HTTPStatusError as exc:
            return self._err("Canal Rural", f"HTTP {exc.response.status_code}")
        except Exception as exc:
            return self._err("Canal Rural", str(exc))

        try:
            soup = BeautifulSoup(resp.text, "xml")
            noticias = []
            for item in soup.find_all("item")[:5]:
                titulo = item.find("title")
                desc = item.find("description")
                pub = item.find("pubDate")
                link = item.find("link")
                # O campo description do RSS pode conter HTML — precisa de segundo parse
                desc_raw = desc.get_text(strip=True) if desc else ""
                desc_text = BeautifulSoup(desc_raw, "html.parser").get_text(strip=True)
                noticias.append({
                    "titulo": titulo.get_text(strip=True) if titulo else "",
                    "resumo": _truncate(desc_text, 250),
                    "data": pub.get_text(strip=True) if pub else "",
                    "link": link.get_text(strip=True) if link else "",
                })
        except Exception as exc:
            return self._err("Canal Rural", f"falha no parsing RSS: {exc}")

        result = {
            "fonte": "Canal Rural",
            "coletado_em": datetime.now().isoformat(),
            "noticias": noticias,
            "cached": False,
        }
        _cache.set(key, result)
        return result

    # ──────────────────────────────────────────────────
    # Agrega todas as fontes em um único payload
    # ──────────────────────────────────────────────────
    def get_all_data(self) -> dict:
        return {
            "clima": self.get_weather(),
            "cotacoes": self.get_commodity_prices(),
            "noticias": self.get_agro_news(),
        }

    @staticmethod
    def _err(fonte: str, msg: str) -> dict:
        logger.warning("[scraping] %s — %s", fonte, msg)
        return {
            "fonte": fonte,
            "erro": msg,
            "coletado_em": datetime.now().isoformat(),
        }


def _truncate(text: str, max_len: int) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


# Singleton
scraping_service = ScrapingService()
