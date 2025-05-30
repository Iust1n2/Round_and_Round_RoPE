import itertools
from data_template import succession_dataset


def generate_all_sequences(dataset):
    sequences = []

    # Step 1: Ascending and Descending sequences
    for values in dataset.values():
        for length in range(6, 11):
            if len(values) >= length:
                for i in range(len(values) - length + 1):
                    slice_seq = values[i : i + length]
                    sequences.append(slice_seq)  # Ascending
                    sequences.append(list(reversed(slice_seq)))  # Descending

    # Step 2: Alternating sequences (every combination with every other)
    keys = list(dataset.keys())
    for key1 in keys:
        for key2 in keys:
            if key1 == key2:
                continue
            values1 = dataset[key1]
            values2 = dataset[key2]
            min_len = min(len(values1), len(values2))
            max_possible = min(min_len, 10)
            for length in range(6, max_possible + 1):
                for i in range(min_len - length + 1):
                    seq1 = values1[i : i + length]
                    seq2 = values2[i : i + length]
                    alternating = list(itertools.chain.from_iterable(zip(seq1, seq2)))
                    sequences.append(alternating)

    return sequences


all_sequences = generate_all_sequences(succession_dataset)

output_file = "training_data.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for seq in all_sequences:
        if len(seq) < 2:
            continue  # Skip sequences too short to split

        # Format for "next term"
        input_seq = seq[:-1]
        answer = seq[-1]
        input_str = " ".join(input_seq)
        f.write(f"the next term in the sequence {input_str} is {answer}\n")

        # Format for "last term"
        full_str = " ".join(seq)
        f.write(f"the last term in the sequence {full_str} is {seq[-1]}\n")
