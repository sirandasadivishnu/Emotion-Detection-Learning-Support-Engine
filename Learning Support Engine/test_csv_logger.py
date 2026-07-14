from services.model_cache import get_bilstm
from services.mixed_emotion import MixedEmotionDetector
from services.prediction_schema import PredictionSchema
from services.csv_logger import CSVLogger

classifier = get_bilstm()

detector = MixedEmotionDetector()

logger = CSVLogger()

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

logger.save(text, result)

print(result)

print("\nSaved to data/prediction_history.csv")