from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


MODEL_NAME = "google/flan-t5-base"


print("Loading local LLM...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

print("Local LLM loaded successfully.")


def generate_answer(
    context,
    question,
    conversation_history=""
):

    prompt = f"""
Read the context carefully and answer the question.

Context:
{context}

Question:
{question}

Write a complete answer in 2 to 4 sentences.

Do not answer with only a keyword or phrase.

Use only information from the context.

If the context does not contain the answer, say:
I could not find this information in the provided document.

Answer:
"""

    print("\n--- GENERATOR DEBUG ---")
    print("Question:", question)
    print("Context length:", len(context))
    print("-----------------------")


    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )


    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        min_new_tokens=20,
        num_beams=5,
        no_repeat_ngram_size=2
    )


    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


    return answer.strip()