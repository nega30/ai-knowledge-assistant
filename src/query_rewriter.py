from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


MODEL_NAME = "google/flan-t5-base"


print("Loading query rewriting model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

print("Query rewriting model loaded.")


def rewrite_query(
    question,
    conversation_history=""
):

    # If there is no conversation history,
    # the original question is already standalone.
    if not conversation_history.strip():
        return question


    prompt = f"""
Rewrite the user's latest question into a
standalone question.

Use the conversation history to understand
references such as:
it, this, that, they, them, those.

Do not answer the question.

Only return the rewritten question.

Conversation history:
{conversation_history}

Latest question:
{question}

Standalone question:
"""


    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )


    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        num_beams=4
    )


    rewritten_question = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


    return rewritten_question.strip()