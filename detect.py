from langdetect import detect

def detect_language(text):
    return detect(text)

"""
text = "Bonjour tout le monde"
language = detect_language(text)
print(f"The detected language is: {language}")
"""