from services.preprocessing import TextPreprocessor

preprocessor = TextPreprocessor()

text = "I am very happy!!! But also a little nervous about tomorrow."

print("=" * 50)
print("Original Text:")
print(text)

print("\n" + "=" * 50)
cleaned = preprocessor.clean_text(text)
print("Cleaned Text:")
print(cleaned)

print("\n" + "=" * 50)
scores = preprocessor.keyword_scores(cleaned)
print("Keyword Scores:")
print(scores)