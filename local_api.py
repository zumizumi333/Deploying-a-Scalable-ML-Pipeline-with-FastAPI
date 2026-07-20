import requests


BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 10

SAMPLE_DATA = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}


def main():
    """Send one GET and one inference POST to the local API."""
    get_response = requests.get(BASE_URL, timeout=REQUEST_TIMEOUT)
    print("GET request")
    print(f"Status Code: {get_response.status_code}")
    get_response.raise_for_status()
    print(f"Result: {get_response.json()}")

    post_response = requests.post(
        f"{BASE_URL}/data/",
        json=SAMPLE_DATA,
        timeout=REQUEST_TIMEOUT,
    )
    print("\nPOST request")
    print(f"Status Code: {post_response.status_code}")
    post_response.raise_for_status()
    print(f"Result: {post_response.json()['result']}")


if __name__ == "__main__":
    main()
