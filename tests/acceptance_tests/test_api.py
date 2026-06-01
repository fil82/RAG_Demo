import os

import httpx
import pytest

pytestmark = pytest.mark.acceptance

BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8080")


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=60) as http_client:
        try:
            http_client.get("/gtg")
        except httpx.HTTPError:
            pytest.skip(f"API not reachable at {BASE_URL}")
        yield http_client


def test_good_to_go(client):
    response = client.get("/gtg")
    assert response.status_code == 200
    assert response.json()["gtg"] == "OK"


def test_retrieve_returns_relevant_chunk(client):
    response = client.post(
        "/retrieve",
        json={"query": "What is the capital of Nepal?", "top_k": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "What is the capital of Nepal?"
    assert len(body["results"]) >= 1
    assert body["answer"]
    top = body["results"][0]
    assert top["score"] is not None
    assert top["metadata"]["document_id"]
