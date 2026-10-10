"""
Integration tests for Basketry FastAPI REST endpoints.
"""

import io
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_api_summary(client):
    res = client.get("/api/summary")
    assert res.status_code == 200
    data = res.json()
    assert "dataset_summary" in data
    assert data["dataset_summary"]["rows"] > 0
    assert data["dataset_summary"]["reviewers"] > 0
    assert "preprocessed_meta" in data


def test_api_mine(client):
    payload = {
        "algorithm": "scratch",
        "min_support": 0.03,
        "min_confidence": 0.35,
        "min_lift": 1.0,
        "min_rating": 4.0,
        "max_len": 3,
    }
    res = client.post("/api/mine", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["algorithm"] == "scratch"
    assert "runtime_ms" in data
    assert "num_itemsets" in data
    assert "num_rules" in data
    assert isinstance(data["rules"], list)
    assert len(data["rules"]) > 0


def test_api_compare(client):
    res = client.get("/api/compare?min_support=0.03&min_confidence=0.35&min_lift=1.0&min_rating=4.0&max_len=3")
    assert res.status_code == 200
    data = res.json()
    assert data["verification"]["itemsets_match"] is True
    assert data["verification"]["rules_match"] is True
    assert "scratch" in data
    assert "apriori" in data
    assert "fpgrowth" in data


def test_api_recommend(client):
    res = client.get("/api/recommend?product=B0001001")
    assert res.status_code == 200
    data = res.json()
    assert "bundles" in data
    assert isinstance(data["bundles"], list)


def test_api_sweep(client):
    res = client.get("/api/sweep?min_rating=4.0")
    assert res.status_code == 200
    data = res.json()
    assert "supports" in data
    assert "confidences" in data
    assert "grid" in data
    assert len(data["grid"]) == len(data["supports"])


def test_api_manual_example(client):
    res = client.get("/api/manual-example")
    assert res.status_code == 200
    data = res.json()
    assert "rule_display" in data
    assert "raw_counts" in data
    assert "metrics" in data
    assert "steps" in data
    assert len(data["steps"]) >= 8


def test_api_upload_csv(client):
    csv_content = (
        "reviewerID,asin,title,rating,timestamp\n"
        "U1,A1,Coffee,5,1600000000\n"
        "U1,A2,Filters,5,1600000001\n"
        "U2,A1,Coffee,4,1600000002\n"
        "U2,A2,Filters,5,1600000003\n"
        "U3,A1,Coffee,5,1600000004\n"
        "U3,A3,Mug,5,1600000005\n"
    )
    files = {"file": ("test_upload.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    res = client.post("/api/upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["source"] == "test_upload.csv"
    assert data["dataset_summary"]["rows"] == 6

    # Restore demo data
    reset_res = client.post("/api/sample?name=demo")
    assert reset_res.status_code == 200
