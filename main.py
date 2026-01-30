from ingest import ingest
from rag import get_related_chunks
from rag import create_prompt
from rag import get_answer_from_llm
from memory import save_qa, load_conversations
from openai import OpenAI
client = OpenAI()
import dotenv
dotenv.load_dotenv()

def user_input():
    user_query = input("User: ")
    if user_query.lower() == "exit" or user_query.lower() == "quit" or user_query.lower() == "stop":
        return None
    return user_query


def create_exact_question(old_conversations, current_question):
    # 1. Filter out the "I don't know" and "break" noise to save tokens
    clean_history = [
        f"User: {c['question']}\nAI: {c['response']}" 
        for c in old_conversations.get("conversations", [])
        if "I don't know" not in c['response'] and c['question'] != 'break'
    ]
    
    # 2. Join into a clean string format
    history_text = "\n".join(clean_history)

    # 3. Use a structured system-style prompt
    prompt = f"""
    Context/History:
    {history_text}

    Current Question: {current_question}

    Task: Based on the history above, generate a concise follow-up question.
    """

    response = client.responses.create(
        model="gpt-5-nano",
        input=prompt
    )

    return response.output_text

        

if __name__ == "__main__":
    ingest()
    while True:
        question = user_input()
        if question is None:
            break

        old_conversations = load_conversations()

        exact_question = create_exact_question(old_conversations, question)

        docs = get_related_chunks(exact_question)

        prompt = create_prompt(docs, question)

        answer = get_answer_from_llm(prompt)

        save_qa(exact_question, answer)

        print("Answer: ", answer)
    
    
    