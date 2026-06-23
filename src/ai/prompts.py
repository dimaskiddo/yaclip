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
    "{language_instruction}\n\n"
    "Return JSON array with EXACTLY {target_clips} items — never fewer, never more. "
    "Fields: candidate_index (int, 1-based), "
    "start_time (float, sec from candidate start), "
    "end_time (float, sec from candidate start), "
    "title (<=50 chars, hook/bait style), "
    "caption (<=150 chars, short hook caption), "
    "description (<=300 chars, hook + context + CTA), "
    "hashtags (string, 5-8 space-separated hashtags like '#gaming #mlbb #shorts'), "
    "content_type (PODCAST|JUST_CHAT|GAMING_SOLO|GAMING_COLLAB), "
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


_CONTENT_TYPE_CONFIDENT_LINE = "Content type: {content_type}. Use for all clips."

_CONTENT_TYPE_UNCERTAIN_LINE = (
    "Content type: UNCERTAIN. Decide per clip from audio+visual+transcript.\n"
    "Rules (priority): 1) Audio ~1 speaker = solo (never collab). "
    "2) Game characters are NOT people — only count webcam faces. "
    "3) Cross-check visual person count against audio.\n"
    "4) Detection evidence block above provides whole-video algorithmic signal — "
    "weigh it above per-clip visual descriptors. "
    "If webcam_count>=2 AND gameplay_present=True, the video is GAMING_COLLAB."
)


def build_language_instruction(language: str) -> str:
    """Build the language + tone + hook/bait instruction block for the LLM prompt."""
    tone_block = (
        "Write titles, captions, and descriptions in a relaxed, informal, human tone — "
        "like a real person posting on social media. Use trending phrases naturally. "
        "Make every title a hook or bait that grabs attention and makes people want to click. "
        "Avoid robotic or corporate AI wording. Avoid hashtags inside the description text.\n"
        "Base ALL generated text on the actual transcript content provided for each candidate. "
        "The title, caption, description, and hashtags must be directly relevant to the clip's "
        "specific spoken content — never generic or made up."
    )
    if language and language.lower() != "auto":
        return (
            f"The video language is {language}. Write all output in {language} only — "
            f"natural and native.\n"
            f"{tone_block}"
        )
    return (
        "Detect the language from the transcript. Write all titles, captions, "
        "and descriptions in that same language — natural and native.\n"
        f"{tone_block}"
    )


def build_detection_evidence_block(evidence: dict[str, object]) -> str:
    """Format detection evidence as a compact keyword-labeled block for the LLM prompt.

    The LLM sees this when the algorithm was uncertain, so it has the raw
    numbers the algorithm used — not just the natural-language descriptors
    from ``_build_descriptor()``.  This prevents hallucination (e.g.
    classifying a gameplay+2-webcam video as PODCAST).

    Example output:
        Detection evidence (whole video):
        gameplay_present=True webcam_count=2 HUD_score=0.0032 gaming_hint=True donation_detected=False open_area_frac=0.52 non_person_motion=8.3
    """
    if not evidence:
        return ""

    # Stable key order for deterministic prompts.
    key_order = [
        "gameplay_present",
        "webcam_count",
        "hud_score",
        "hud_detected",
        "gaming_hint",
        "donation_detected",
        "open_area_frac",
        "non_person_motion",
    ]
    ordered = [(k, evidence[k]) for k in key_order if k in evidence]
    # Append any unexpected keys at the end.
    extra = [(k, evidence[k]) for k in evidence if k not in key_order]
    ordered.extend(extra)

    parts = [f"{k}={v}" for k, v in ordered]
    return "Detection evidence (whole video):\n" + " ".join(parts) + "\n"


def get_system_prompt(
    content_type: str | None = "PODCAST",
    target_duration: int = 60,
    language: str = "auto",
) -> str:
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
    language_instruction = build_language_instruction(language)
    return template.format(
        content_type=content_type or "UNCERTAIN",
        content_type_line=content_type_line,
        language_instruction=language_instruction,
        target_duration=target_duration,
        min_duration=min_duration,
        max_duration=max_duration,
    )


def get_language_prompt(language: str) -> str | None:
    """Resolve a dialect-locking STT prompt for the given language code or name."""
    lang_code = language.lower()
    lang_code = LANGUAGE_MAP.get(lang_code, lang_code)
    return LANGUAGE_PROMPTS.get(lang_code)


def build_batch_user_prompt(candidates_text: str, target_clips: int, base_sys_prompt: str) -> str:
    """Build the user prompt for batch (multi-candidate) LLM clip selection.

    Used by both cloud (OpenAI / Gemini) and local LLM providers to avoid
    duplicating the same prompt template across three files.
    """
    return (
        f"Here are the candidate segments:\n\n{candidates_text}\n\n"
        f"Select EXACTLY {target_clips} clips — no more, no fewer. You MUST return exactly {target_clips} items. "
        "Timestamps are relative to candidate window start (0.0 = start of the candidate's segment).\n"
        f"Return JSON array with fields as specified below.\n"
        f"{base_sys_prompt}\n\n"
        "Format response ONLY as a valid JSON array, no markdown wrappers."
    )


def strip_json_markdown(text: str) -> str:
    """Strip common markdown code fence wrappers from LLM JSON responses."""
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
