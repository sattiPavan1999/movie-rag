import json
import os

CONVERSATION_FILE = "data/conversations.json"


def load_conversations():
    if not os.path.exists(CONVERSATION_FILE):
        return {"conversations": []}

    with open(CONVERSATION_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
    


def save_qa(question: str, response: str):
    data = load_conversations()

    data["conversations"].append(
        {
            "question": question,
            "response": response
        }
    )

    with open(CONVERSATION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
