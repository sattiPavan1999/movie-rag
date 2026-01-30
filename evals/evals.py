import sys
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import get_related_chunks, create_prompt, get_answer_from_llm
from main import create_exact_question

EVAL_TESTS = [
    {
        "name": "single_turn_director",
        "question": "Who directed Interstellar?",
        "expected": "Christopher Nolan"
    },
    {
        "name": "unknown_question",
        "question": "What is the budget of Interstellar?",
        "expected": "I don't know"
    }
]


def run_eval(test):
    print("question: ", test["question"])
    related_chunks = get_related_chunks(test["question"])
    print("related_chunks: ", related_chunks)



def run_all_evals():
    print("\nRunning Evals...\n")
    passed_count = 0

    for test in EVAL_TESTS:
        run_eval(test)

        # status = "✅ PASS" if result["passed"] else "❌ FAIL"
        # print(f"{status} | {result['name']}")
        # print(f"Expected: {result['expected']}")
        # print(f"Actual:   {result['actual']}\n")

        # if result["passed"]:
        #     passed_count += 1

    print(f"Summary: {passed_count}/{len(EVAL_TESTS)} tests passed")


if __name__ == "__main__":
    run_all_evals()
