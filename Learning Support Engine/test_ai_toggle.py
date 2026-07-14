from services.gemini_service import get_gemini_response
from services.response_templates import EMOTION_RESPONSES

field = "Computer Science"
problem = "I don't understand Binary Search."
emotion = "fear"
confidence = 0.91

USE_AI = False

if USE_AI:
    response = get_gemini_response(
        field,
        problem,
        emotion,
        confidence
    )
else:
    template = EMOTION_RESPONSES[emotion]

    response = (
        f"{template['emoji']}\n"
        f"{template['response']}\n\n"
        f"Suggestion: {template['action']}"
    )

print("\n========== RESPONSE ==========\n")
print(response)