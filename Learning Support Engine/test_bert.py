from services.bert_model import BERTClassifier

classifier = BERTClassifier()

text = "I am extremely happy because I got selected."

result = classifier.predict(text)

print("\nPrediction")
print(result)