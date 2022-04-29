import requests


NEWS_API_SKILL_URL = "http://0.0.0.0:8066/respond"

topic = "sport"
dialogs = {
    "dialogs": [
        {
            "utterances": [],
            "bot_utterances": [],
            "human": {"attributes": {}},
            "human_utterances": [
                {
                    "text": f"news about {topic}",
                    "annotations": {
                        "ner": [[{"text": topic}]],
                        "cobot_topics": {"text": ["News"]},
                        "news_api_annotator": {
                            "entity": topic,
                            "news": {"title": "Some News about Sport.", "content": "Some News about Sport."},
                        },
                    },
                }
            ],
        }
    ]
}

result = requests.post(NEWS_API_SKILL_URL, json=dialogs, timeout=1.5)
result = result.json()

assert result[0][1] == 1.0 and result[0][-1]["news_topic"] == "sport", print(result)

print("SUCCESS")
