import random

from student import tokenizer

random.seed(67)


def main():
    tok = tokenizer.Tokenizer.from_files("artifacts/tinystories_vocab.pkl", "artifacts/tinystories_merges.pkl", ["<|endoftext|>"])

    with open("data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as file:
        text = file.read()

    documents = text.split("<|endoftext|>")
    documents = [doc for doc in documents if doc.strip()]
    sampled_documents = random.sample(documents, 10)

    total_num_bytes = 0
    total_num_tokens = 0
    for doc in sampled_documents:
        total_num_bytes += len(doc.encode("utf-8"))
        token_ids = tok.encode(doc)
        total_num_tokens += len(token_ids)

    print(total_num_bytes/total_num_tokens)


if __name__ == "__main__":
    main()





