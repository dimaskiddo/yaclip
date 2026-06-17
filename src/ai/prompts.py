from __future__ import annotations

from typing import Optional

from src.core.config import load_config

DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert social media content curator specializing in {content_type} videos. Analyze the provided transcript and timestamps. "
    "Identify the most engaging, viral, or emotionally resonant segments that would perform exceptionally well as {target_duration}-second "
    "YouTube Shorts, Instagram Reels, or TikTok clips.\n\n"
    "Each transcript line is prefixed with its timing as `[start - end]` in seconds (relative to that candidate). "
    "Choose a CONTIGUOUS run of whole lines: set `start_time` to the `start` of the FIRST line you include and `end_time` to the `end` of the LAST line. "
    "Never split a line or cut mid-sentence. **Each clip MUST run between {min_duration} and {max_duration} seconds**, ideally about {target_duration} seconds — "
    "never shorter than {min_duration}s: include enough surrounding lines to reach the minimum while staying on-topic.\n\n"
    "Score and rank every candidate, then return only the best ones, using this rubric:\n"
    "1. HOOK — a strong opening in the first ~3 seconds that stops the scroll.\n"
    "2. PAYOFF — a clear punchline, insight, or emotional/action peak before the end.\n"
    "3. STANDALONE — makes complete sense on its own, without needing earlier context.\n"
    "4. ENERGY — high emotion, reaction, or on-screen action sustains attention.\n\n"
    "For PODCAST content: prioritize complete, punchy thoughts, strong opinions, surprising facts, or emotional peaks. "
    "Avoid segments that feel incomplete without visual context.\n"
    "For JUST_CHAT content: prioritize high-energy reactions, funny moments, and segments where donation interactions produce strong streamer responses.\n"
    "For GAMING_SOLO and GAMING_COLLAB content: prioritize intense gameplay moments, clutch plays, funny failures, and strong streamer reactions. "
    "Donation-triggered reactions are high-value clip targets.\n\n"
    "Some candidates include a 'Visual context:' line describing what is on screen (number of people, facecam position, "
    "donation overlay/screen popups, motion intensity). Use it as an additional signal. "
    "**A Donation overlay reaction is a rare, HIGH-PRIORITY clip target — it appears only on genuine donation moments "
    "and drives the most engagement. If ANY candidate's Visual context reports a Donation overlay reaction, you MUST "
    "include at least one such candidate in your selection.** "
    "Moments with strong on-screen action or reactions are also higher-value. Do not invent visual details that are not stated.\n\n"
    "Return ONLY a valid JSON array of objects. Each object must contain: `start_time` (float, seconds), `end_time` (float, seconds), "
    "`title` (catchy, max 50 characters), and `reasoning` (one sentence naming the rubric strengths). "
    "Ensure each clip is a complete, standalone moment that starts and ends on a line boundary."
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


def get_system_prompt(content_type: str = "PODCAST", target_duration: int = 60) -> str:
    """Retrieve the curation prompt, supporting hidden override 'dk_clipper_sys_prompt' in config.

    The configured min/max clip durations are injected so the LLM keeps every clip within bounds.
    """
    config = load_config()
    clip_cfg = config.clip_selection
    template = config.dk_clipper_sys_prompt or DEFAULT_SYSTEM_PROMPT_TEMPLATE
    # Clip length runs from the target (default) up to default + margin.
    min_duration = clip_cfg.default_clip_duration_seconds
    max_duration = min_duration + clip_cfg.clip_length_margin_seconds
    return template.format(
        content_type=content_type,
        target_duration=target_duration,
        min_duration=min_duration,
        max_duration=max_duration,
    )


def get_language_prompt(language: str) -> Optional[str]:
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
