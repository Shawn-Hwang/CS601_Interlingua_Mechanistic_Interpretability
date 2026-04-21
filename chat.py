"""Simple stateless chat loop using the project's loaded model."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from utils import load_model, generate_text

MAX_NEW_TOKENS = 500


def main():
    model_name = "gemma-3-4b-it"
    model = load_model(model_name)

    print("\nModel loaded. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() == "quit":
            break

        if not user_input:
            continue

        response = generate_text(model, user_input, max_new_tokens=MAX_NEW_TOKENS)
        print(f"Model: {response}\n")


if __name__ == "__main__":
    main()
