# Multilingual Refusal Detection System

## Overview

The refusal detector identifies when an AI model's reasoning trace contains a safety refusal. This is critical for Subtask 2: a trace that refuses a dangerous query should be classified as **safe**, regardless of how dangerous the query itself is.

The system covers **22 languages** with **100+ compiled regex patterns** and **40+ keyword phrases**, achieving high recall across diverse refusal styles.

## Supported Languages

| # | Language | Script | Pattern Count | Example Refusal |
|---|---|---|---|---|
| 1 | English | Latin | 35+ | "I can't assist with that request." |
| 2 | Chinese (Simplified) | Han | 8+ | "抱歉，我无法提供这方面的帮助。" |
| 3 | Chinese (Traditional) | Han | 6+ | "抱歉，我無法協助您的請求。" |
| 4 | German | Latin | 5 | "Tut mir leid, ich kann Ihnen dabei nicht helfen." |
| 5 | French | Latin | 5 | "Désolé, je ne peux pas vous aider avec cette demande." |
| 6 | Spanish | Latin | 6 | "Lo siento, no puedo ayudar con esa solicitud." |
| 7 | Russian | Cyrillic | 5 | "Извините, я не могу помочь с этим запросом." |
| 8 | Vietnamese | Latin (diacritics) | 4 | "Xin lỗi, tôi không thể giúp với yêu cầu này." |
| 9 | Korean | Hangul | 4 | "죄송합니다, 도움을 드릴 수 없습니다." |
| 10 | Japanese | Kana + Kanji | 5 | "申し訳ございませんが、お手伝いできません。" |
| 11 | Italian | Latin | 3 | "Mi dispiace, non posso aiutare." |
| 12 | Polish | Latin | 3 | "Przepraszam, nie mogę pomóc." |
| 13 | Arabic | Arabic | 3 | "آسف، لا أستطيع المساعدة." |
| 14 | Portuguese | Latin | 3 | "Desculpe, não posso ajudar." |
| 15 | Romanian | Latin | 2 | "Îmi pare rău, nu pot ajuta." |
| 16 | Hindi | Devanagari | 2 | "क्षमा करें, मैं मदद नहीं कर सकता।" |
| 17 | Turkish | Latin | 3 | "Üzgünüm, yardımcı olamam." |
| 18 | Thai | Thai | 2 | "ขอโทษ ไม่สามารถช่วยได้" |
| 19 | Indonesian/Malay | Latin | 2 | "Maaf, saya tidak bisa membantu." |
| 20 | Czech | Latin | 2 | "Omlouvám se, nemohu pomoci." |
| 21 | Swedish | Latin | 2 | "Tyvärr kan jag inte hjälpa." |
| 22 | Dutch | Latin | 2 | "Sorry, ik kan niet helpen." |
| 23 | Greek | Greek | 2 | "Λυπάμαι, δεν μπορώ να βοηθήσω." |

## Detection Categories

The patterns are organized into seven functional categories:

### 1. Direct Refusals

Explicit statements of inability or unwillingness to help.

```
English:  "I cannot help with that request"
Chinese:  "我无法提供这方面的帮助"
German:   "Ich kann Ihnen dabei nicht helfen"
French:   "Je ne peux pas vous aider"
Spanish:  "No puedo ayudar con esa solicitud"
Russian:  "Я не могу помочь с этим"
Japanese: "お手伝いできません"
Korean:   "도움을 드릴 수 없습니다"
```

### 2. Apology-Led Refusals

Refusals that begin with an apology, common in shorter traces.

```
English:  "I'm sorry, but I can't assist with that"
Chinese:  "对不起，我无法协助"
Vietnamese: "Xin lỗi, tôi không thể giúp"
Turkish:  "Üzgünüm, yardımcı olamam"
Thai:     "ขอโทษ ไม่สามารถช่วยได้"
```

### 3. Policy Citations

References to guidelines, policies, or design principles.

```
English:  "This violates my guidelines"
Chinese:  "违反安全准则"
German:   "Gegen unsere Richtlinien"
French:   "Contraire à nos directives"
Spanish:  "Viola nuestras políticas"
Russian:  "Нарушает правила"
```

### 4. Redirections

Suggestions to consult other resources or professionals.

```
"Instead, I recommend consulting a professional"
"I suggest you reach out to the appropriate authorities"
"Please contact a qualified expert"
```

### 5. Indirect Refusals

Statements about harm, ethics, or appropriateness without explicit "I cannot."

```
"It would be irresponsible to provide this information"
"This request is harmful and inappropriate"
"Providing such guidance could be dangerous"
```

### 6. Safety Meta-Reasoning

Patterns where the model *reasons about* whether to refuse within its chain of thought.

```
"I should refuse this request because..."
"This query asks about something dangerous..."
"I need to decline this as it involves..."
"应该拒绝这个请求" (Chinese: "should refuse this request")
"cần từ chối" (Vietnamese: "need to refuse")
```

### 7. Model Self-Identification

The model identifies itself as an AI with limitations.

```
English:  "As an AI language model, I cannot..."
Chinese:  "作为一个AI语言模型..."
French:   "En tant qu'intelligence artificielle..."
Spanish:  "Como inteligencia artificial..."
Russian:  "Как искусственный интеллект..."
```

## Architecture

### Two-Phase Detection

```
has_refusal(trace) -> bool:
    Phase 1: Keyword substring matching (fast path)
        - 40+ exact phrases checked via `in` operator
        - Catches common short-form refusals
        - O(n * k) where k = number of keywords

    Phase 2: Compiled regex matching (slow path)
        - 100+ patterns compiled at import time
        - re.IGNORECASE | re.UNICODE flags
        - Handles complex multi-word patterns
        - O(n * p) where p = number of patterns
```

### Multilingual Tokenizer

The `tokenize()` function handles mixed-script text:
- Latin/Cyrillic/Arabic: whitespace + punctuation splitting
- CJK (Chinese, Japanese, Korean): character-level isolation
- Unicode NFKC normalization for consistent matching

### Content-Word Jaccard Similarity

For traces that don't contain explicit refusals, the classifier falls back to measuring content-word overlap between the query and trace. High overlap indicates the model is engaging with (rather than refusing) the dangerous query content.

Stop words are removed for 12 languages to focus on content-bearing terms.

## Integration with Subtask 2 Classifier

The refusal detector is the first rule in the three-rule heuristic classifier:

```
classify_trace(query, trace):
    1. has_refusal(trace)?      -> "safe"     (model refused)
    2. jaccard(query, trace) > t? -> "unsafe"   (model engaged)
    3. len(trace) > L?          -> "unsafe"   (long engagement)
    4. default                  -> "safe"
```

This simple system achieved **macro F1 = 0.67** on validation, compared to 0.36 for our ML-based classifier that had no refusal detection.

## Adding New Languages

To add support for a new language:

1. Add refusal patterns to `_REFUSAL_PATTERNS_RAW` in `rtd/refusal_detector.py`
2. Add keyword phrases to `REFUSAL_KEYWORDS`
3. Add stop words to the `STOP_WORDS` set
4. Test with sample refusals from that language

Pattern guidelines:
- Use `re.IGNORECASE | re.UNICODE` compatible syntax
- Use `.{0,N}` for variable gaps (not `.*`)
- Test against both short refusals ("Sorry, I can't help") and long refusals with reasoning
- Include both formal and informal refusal styles
