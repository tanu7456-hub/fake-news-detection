import ipaddress
import os
import socket
from urllib.parse import urlparse

import joblib
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request

from fact_checker import search_fact_checks

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "model", "fake_news_model.joblib"
)

# The fact-checking website can run even if the optional ML
# model has not been trained yet.
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        model = None


def validate_public_url(url):
    """Allow only public HTTP/HTTPS URLs."""
    try:
        parsed = urlparse(url.strip())

        if parsed.scheme not in ("http", "https"):
            return False

        if not parsed.hostname or parsed.username or parsed.password:
            return False

        host = parsed.hostname

        # Reject local hostnames.
        if host.lower() == "localhost" or host.endswith(".local"):
            return False

        # Resolve the hostname and reject private/reserved IPs.
        addresses = socket.getaddrinfo(
            host, parsed.port or (443 if parsed.scheme == "https" else 80)
        )

        if not addresses:
            return False

        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if not ip.is_global:
                return False

        return True

    except (ValueError, OSError, socket.gaierror):
        return False


def extract_article(url):
    """Fetch a public web page and extract readable text."""
    if not validate_public_url(url):
        raise ValueError(
            "Invalid URL. Enter a public HTTP or HTTPS article URL."
        )

    headers = {
        "User-Agent": "NewsFactCheckStudentProject/1.0"
    }

    try:
        with requests.get(
            url.strip(),
            headers=headers,
            timeout=(5, 10),
            stream=True,
            allow_redirects=False,
        ) as response:

            # Do not follow redirects to unvalidated destinations.
            if response.is_redirect or response.is_permanent_redirect:
                raise ValueError(
                    "This URL redirects. Please enter the final article URL."
                )

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            if "text/html" not in content_type:
                raise ValueError(
                    "This URL does not appear to be an HTML article."
                )

            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > 2_000_000:
                raise ValueError("The article page is too large.")

            chunks = []
            total = 0

            for chunk in response.iter_content(chunk_size=8192):
                total += len(chunk)

                if total > 2_000_000:
                    raise ValueError("The article page is too large.")

                chunks.append(chunk)

            html = b"".join(chunks)

    except requests.RequestException as exc:
        raise ValueError(
            "Could not fetch this article. Check the URL or paste its text."
        ) from exc

    soup = BeautifulSoup(html, "html.parser")

    for element in soup([
        "script", "style", "nav", "footer",
        "header", "aside", "noscript"
    ]):
        element.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    paragraphs = [
        p.get_text(" ", strip=True)
        for p in soup.find_all("p")
    ]
    article_text = " ".join(p for p in paragraphs if p)

    if len(article_text) < 50:
        raise ValueError(
            "Could not extract enough article text. "
            "Paste the news text manually instead."
        )

    return title, article_text[:20000]


def optional_ml_prediction(text):
    """Supporting model signal only; not factual verification."""
    if model is None:
        return None

    try:
        prediction = int(model.predict([text])[0])

        labels = {
            0: "FAKE",
            1: "REAL",
        }

        result = {
            "label": labels.get(prediction, str(prediction)),
            "confidence": None,
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([text])[0]
            result["confidence"] = round(
                float(max(probabilities)) * 100, 1
            )

        return result

    except Exception:
        return None


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    news_text = ""
    news_url = ""
    article_title = ""
    error = None

    if request.method == "POST":
        news_text = request.form.get("news_text", "").strip()
        news_url = request.form.get("news_url", "").strip()

        if not news_text and not news_url:
            error = "Please paste news text, enter a URL, or provide both."

        elif len(news_text) > 20000:
            error = "News text must be 20,000 characters or less."

        else:
            article_text = news_text

            if news_url:
                try:
                    article_title, extracted_text = extract_article(news_url)

                    # If the user pasted text too, include both.
                    if news_text:
                        article_text = (
                            news_text + "\n\n" + extracted_text
                        )[:20000]
                    else:
                        article_text = extracted_text

                except ValueError as exc:
                    error = str(exc)

            if not error:
                # Search the submitted claim/article in published
                # fact-check reports.
                fact_result = search_fact_checks(article_text[:500])

                result = {
                    "status": fact_result.get("status", "Uncertain"),
                    "message": fact_result.get(
                        "message", "Evidence could not be assessed."
                    ),
                    "sources": fact_result.get("sources", []),
                    "ml": optional_ml_prediction(article_text),
                    "article_title": article_title,
                }

                # A found report is not automatically a matching verdict.
                # Only publisher evidence is displayed; no automatic
                # TRUE/FALSE conclusion is inferred from search presence.

    return render_template(
        "index.html",
        result=result,
        news_text=news_text,
        news_url=news_url,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
