import requests
import os


SERVICE_PORT = os.getenv("SERVICE_PORT")
URL = f"http://0.0.0.0:{SERVICE_PORT}"
MODEL_URL = f"{URL}/model"
ANNOTATION_URL = f"{URL}/annotation"


def run_test():
    model_test_data = ["Reply.Acknowledge"]
    model_hypothesis = requests.post(MODEL_URL, json=model_test_data).json()

    print("test name: sfp model_hypothesis")
    assert model_hypothesis == [{}]

    annotation_test_data = ["Reply.Acknowledge"]
    annotation_hypothesis = requests.post(ANNOTATION_URL, json=annotation_test_data).json()

    print("test name: sfp annotation_hypothesis")
    assert annotation_hypothesis == [{"batch": [{}]}]

    print("Success")


if __name__ == "__main__":
    run_test()
