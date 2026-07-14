from services.gemini_service import get_gemini_response

response = get_gemini_response(
    field="Computer Science",
    problem="I don't understand Binary Search.",
    emotion="Confused",
    confidence=0.91
)

print(response)
