import translators as ts

def translate_text(text, source_lang='auto', target_lang='en'):
    translation = ts.translate_text(text, translator="bing", from_language=source_lang ,to_language=target_lang)
    return translation

