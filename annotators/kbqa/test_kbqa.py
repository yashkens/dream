import requests


def main():
    url = "http://0.0.0.0:8072/model"

    request_data = [
        {"x_init": ["Who is Donald Trump?"], "entities": [["Donald Trump"]]},
        {"x_init": ["How old is Donald Trump?"], "entities": [["Donald Trump"]]},
    ]

    gold_answers = ["Donald Trump is 45th and current president of the United States.", "Donald Trump is 75 years old."]
    count = 0
    for data, gold_ans in zip(request_data, gold_answers):
        result = requests.post(url, json=data).json()
        res_ans = result[0]["answer"]
        if res_ans == gold_ans:
            count += 1
        else:
            print(f"Got {res_ans}, but expected: {gold_ans}")

    if count == len(request_data):
        print("Success")


if __name__ == "__main__":
    main()
