from services.bilstm_model import BiLSTMClassifier

classifier = BiLSTMClassifier()

text = "I am extremely happy because I got selected."

result = classifier.predict(text)

print("\nPrediction")
print(result)