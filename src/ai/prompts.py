from __future__ import annotations

from src.core.config import load_config

DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
    "You are a short-form clip curator. Pick the best {target_duration}s segments "
    "from the transcript for social media shorts.\n\n"
    "Lines: `[start - end] text`. Pick contiguous whole lines only.\n"
    "Clip duration: {min_duration}s–{max_duration}s (target {target_duration}s).\n\n"
    "Rank by: 1) HOOK (strong first 3s), 2) PAYOFF (punchline/peak), "
    "3) STANDALONE (self-contained), 4) ENERGY (emotion/action).\n\n"
    "PODCAST: punchy thoughts, strong opinions, emotional peaks.\n"
    "JUST_CHAT: reactions, funny moments, donations.\n"
    "GAMING: clutch plays, funny failures, streamer reactions. Donation = top priority.\n\n"
    "Visual context describes on-screen activity. Donation overlay = MUST include at least one. "
    "Strong action = higher value. Never invent details.\n\n"
    "{content_type_line}\n\n"
    "Return JSON array. Fields: candidate_index (int, 1-based), start_time (float s), end_time (float s), "
    "title (<=50 chars), content_type (PODCAST|JUST_CHAT|GAMING_SOLO|GAMING_COLLAB), "
    "reasoning (one sentence)."
)

# Native-language "here is an accurate transcription" primer per ISO 639-1 code. Passed to Whisper /
# OpenAI / Gemini as the language-locking initial prompt when an explicit language is configured.
LANGUAGE_PROMPTS = {
    "id": "Berikut adalah transkripsi yang akurat dan jelas dalam bahasa Indonesia:",
    "en": "Here is a clear and accurate transcription of the audio, with proper punctuation:",
    "es": "A continuación se muestra una transcripción clara y precisa del audio:",
    "fr": "Voici une transcription claire et précise de l'audio:",
    "de": "Hier ist eine klare und genaue Transkription des Audios:",
    "ja": "以下は、音声の明確で正確な文字起こしです：",
    "ko": "다음은 오디오의 명확하고 정확한 전사입니다:",
    "zh": "以下是音频的清晰准确的转录：",
    "pt": "A seguir está uma transcrição clara e precisa do áudio, com pontuação adequada:",
    "it": "Di seguito una trascrizione chiara e accurata dell'audio, con punteggiatura corretta:",
    "ru": "Ниже приведена точная и понятная расшифровка аудио с правильной пунктуацией:",
    "ar": "فيما يلي تفريغ دقيق وواضح للمقطع الصوتي مع علامات الترقيم الصحيحة:",
    "hi": "यहाँ ऑडियो का स्पष्ट और सटीक प्रतिलेखन उचित विराम चिह्नों के साथ दिया गया है:",
    "tr": "Aşağıda sesin doğru noktalama işaretleriyle net ve doğru bir dökümü yer almaktadır:",
    "vi": "Sau đây là bản ghi âm rõ ràng và chính xác của đoạn âm thanh, có dấu câu đầy đủ:",
    "th": "ต่อไปนี้คือการถอดเสียงที่ชัดเจนและถูกต้องของไฟล์เสียง พร้อมเครื่องหมายวรรคตอนที่ถูกต้อง:",
    "pl": "Poniżej znajduje się dokładna i czytelna transkrypcja nagrania z poprawną interpunkcją:",
    "nl": "Hieronder staat een duidelijke en nauwkeurige transcriptie van de audio, met correcte interpunctie:",
    "sv": "Här är en tydlig och korrekt transkription av ljudet, med rätt skiljetecken:",
    "no": "Her er en tydelig og nøyaktig transkripsjon av lyden, med riktig tegnsetting:",
    "da": "Her er en klar og nøjagtig transskription af lyden med korrekt tegnsætning:",
    "fi": "Tässä on selkeä ja tarkka litterointi äänitteestä oikeilla välimerkeillä:",
    "el": "Ακολουθεί μια σαφής και ακριβής μεταγραφή του ήχου, με σωστή στίξη:",
    "he": "להלן תמלול ברור ומדויק של האודיו, עם פיסוק נכון:",
    "uk": "Нижче наведено точну та зрозумілу транскрипцію аудіо з правильною пунктуацією:",
    "ms": "Berikut ialah transkripsi yang jelas dan tepat bagi audio, dengan tanda baca yang betul:",
    "tl": "Narito ang isang malinaw at tumpak na transkripsiyon ng audio, na may wastong bantas:",
    "bn": "নিচে অডিওটির একটি স্পষ্ট ও নির্ভুল প্রতিলিপি যথাযথ যতিচিহ্নসহ দেওয়া হলো:",
    "ta": "ஒலியின் தெளிவான மற்றும் துல்லியமான படியெடுப்பு சரியான நிறுத்தற்குறிகளுடன் கீழே:",
    "ur": "ذیل میں آڈیو کی ایک واضح اور درست نقل درست اوقاف کے ساتھ دی گئی ہے:",
    "fa": "در زیر رونویسی دقیق و واضح صدا با نشانه‌گذاری درست آمده است:",
    "ro": "Mai jos este o transcriere clară și exactă a materialului audio, cu punctuație corectă:",
    "cs": "Níže je uveden jasný a přesný přepis zvuku se správnou interpunkcí:",
    "hu": "Az alábbiakban a hanganyag világos és pontos átirata olvasható, helyes központozással:",
}

# Spoken-language NAME → ISO 639-1 code, so users can configure either form.
LANGUAGE_MAP = {
    "indonesian": "id",
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "japanese": "ja",
    "korean": "ko",
    "chinese": "zh",
    "mandarin": "zh",
    "portuguese": "pt",
    "italian": "it",
    "russian": "ru",
    "arabic": "ar",
    "hindi": "hi",
    "turkish": "tr",
    "vietnamese": "vi",
    "thai": "th",
    "polish": "pl",
    "dutch": "nl",
    "swedish": "sv",
    "norwegian": "no",
    "danish": "da",
    "finnish": "fi",
    "greek": "el",
    "hebrew": "he",
    "ukrainian": "uk",
    "malay": "ms",
    "filipino": "tl",
    "tagalog": "tl",
    "bengali": "bn",
    "tamil": "ta",
    "urdu": "ur",
    "persian": "fa",
    "farsi": "fa",
    "romanian": "ro",
    "czech": "cs",
    "hungarian": "hu",
}


_CONTENT_TYPE_CONFIDENT_LINE = (
    "Content type: {content_type}. Use for all clips."
)

_CONTENT_TYPE_UNCERTAIN_LINE = (
    "Content type: UNCERTAIN. Decide per clip from audio+visual+transcript.\n"
    "Rules (priority): 1) Audio ~1 speaker = solo (never collab). "
    "2) Game characters are NOT people — only count webcam faces. "
    "3) Cross-check visual person count against audio."
)


def get_system_prompt(content_type: str | None = "PODCAST", target_duration: int = 60) -> str:
    """Retrieve the curation prompt, supporting hidden override 'dk_clipper_sys_prompt' in config.

    The configured min/max clip durations are injected so the LLM keeps every clip within bounds.
    """
    config = load_config()
    clip_cfg = config.clip_selection
    template = config.dk_clipper_sys_prompt or DEFAULT_SYSTEM_PROMPT_TEMPLATE
    # Clip length runs from the target (default) up to default + margin.
    min_duration = clip_cfg.default_clip_duration_seconds
    max_duration = min_duration + clip_cfg.clip_length_margin_seconds
    # Content type instruction: confident → echo detected type; uncertain → LLM decides per clip.
    if content_type is not None:
        content_type_line = _CONTENT_TYPE_CONFIDENT_LINE.format(content_type=content_type)
    else:
        content_type_line = _CONTENT_TYPE_UNCERTAIN_LINE
    return template.format(
        content_type=content_type or "UNCERTAIN",
        content_type_line=content_type_line,
        target_duration=target_duration,
        min_duration=min_duration,
        max_duration=max_duration,
    )


def get_language_prompt(language: str) -> str | None:
    """Resolve a dialect-locking STT prompt for the given language code or name."""
    lang_code = language.lower()
    lang_code = LANGUAGE_MAP.get(lang_code, lang_code)
    return LANGUAGE_PROMPTS.get(lang_code)


def strip_json_markdown(text: str) -> str:
    """Strip common markdown code fence wrappers from LLM JSON responses."""
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
