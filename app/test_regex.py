import re

def extract_tags(text):
    # Updated regex
    text = re.sub(r'[^\w\s]+', ' ', text.lower())
    return text.split()

german_text = "Ein schönes Gebäude mit großen Fenstern und einer Tür."
tags = extract_tags(german_text)
print(f"Original: {german_text}")
print(f"Tags: {tags}")
