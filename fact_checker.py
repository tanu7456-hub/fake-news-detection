import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY")

API_URL = (
    "https://factchecktools.googleapis.com/"
    "v1alpha1/claims:search"
)


def search_fact_checks(text):
    if not API_KEY:
        return {
            "status": "Uncertain",
            "message": "Fact-check API key is not configured.",
            "sources": []
        }

    query = " ".join(text.split())[:500]

    try:
        response = requests.get(
            API_URL,
            params={
                "query": query,
                "languageCode": "en",
                "pageSize": 10,
                "key": API_KEY,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

    except requests.RequestException:
        return {
            "status": "Uncertain",
            "message": "Could not retrieve fact-check evidence.",
            "sources": []
        }

    claims = data.get("claims", [])
    sources = []

    for claim in claims:
        claim_text = claim.get("text", "Claim not provided")

        for review in claim.get("claimReview", []):
            sources.append({
                "claim": claim_text,
                "publisher": review.get(
                    "publisher", {}
                ).get("name", "Unknown publisher"),
                "title": review.get(
                    "title", "Fact-check report"
                ),
                "rating": review.get(
                    "textualRating", "Rating not provided"
                ),
                "url": review.get("url", ""),
                "date": review.get("reviewDate", ""),
            })

    if not sources:
        return {
            "status": "Uncertain",
            "message": (
                "No matching published fact-check was found. "
                "This does not prove the news is true or false."
            ),
            "sources": []
        }

    return {
        "status": "Fact-checks found",
        "message": (
            "Read the publishers' verdicts and compare "
            "the reviewed claims with the submitted news."
        ),
        "sources": sources[:10]
    }
