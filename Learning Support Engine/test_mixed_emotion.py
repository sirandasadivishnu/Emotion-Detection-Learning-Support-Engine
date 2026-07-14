from services.bilstm_model import BiLSTMClassifier
from services.mixed_emotion import MixedEmotionDetector
from services.prediction_schema import PredictionSchema

classifier = BiLSTMClassifier()

detector = MixedEmotionDetector()

text = "I am extremely happy because I got selected."

prediction = classifier.predict(text)

mixed = detector.detect(prediction)

result = PredictionSchema.build(

    model_name="BiLSTM",

    cleaned_text=mixed["cleaned_text"],

    primary_emotion=mixed["primary_emotion"],

    primary_confidence=mixed["primary_confidence"],

    secondary_emotions=mixed["secondary_emotions"],

    scores=mixed["scores"]

)

print(result)