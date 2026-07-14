from services.session_manager import SessionManager
from services.interaction_logger import InteractionLogger

session = SessionManager()

logger = InteractionLogger()

interaction = {

    "field": "Computer Science",

    "problem": "I don't understand Binary Search.",

    "emotion": "fear",

    "confidence": 0.91,

    "response": "Practice Binary Search step by step."

}

session.add(interaction)

logger.save(

    field=interaction["field"],

    problem=interaction["problem"],

    emotion=interaction["emotion"],

    confidence=interaction["confidence"],

    ai_enabled=True,

    response=interaction["response"]

)

print("\nSession History\n")

print(session.get_history())

print("\nSaved to data/interaction_history.csv")