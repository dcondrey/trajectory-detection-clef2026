"""
Heuristic classifier for PAN-CLEF 2026 Subtask 2 (Safety Classification).

Achieves ~F1=0.67+ on validation using three rules:
  1. Multilingual refusal detection -> safe
  2. Content-word Jaccard(query, trace) > threshold -> unsafe
  3. Trace length > threshold -> unsafe
  4. Default -> safe

Plus step-level detailed_labels based on per-sentence safety signals.
"""

import csv
import json
import logging
import os
import re
import sys
import unicodedata
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "data" / "subtask2"
TEST_FILE = DATA_DIR / "test" / "subtask2_test.jsonl"
VAL_DIR = DATA_DIR / "validation"
OUTPUT_DIR = PROJECT / "codabench_submissions" / "subtask2"

# ---------------------------------------------------------------------------
# Thresholds (tuned on validation)
# ---------------------------------------------------------------------------
JACCARD_THRESHOLD = 0.080
LENGTH_THRESHOLD = 3500

# ---------------------------------------------------------------------------
# Dangerous topic keywords in queries (multilingual)
# ---------------------------------------------------------------------------
QUERY_DANGER_KEYWORDS = [
    # English
    "hack", "hacking", "exploit", "attack", "malware", "phishing", "inject",
    "vulnerability", "bypass", "crack", "steal", "weapon", "bomb", "drug",
    "kill", "murder", "harm", "abuse", "fraud", "scam", "ransomware", "ddos",
    "trojan", "keylogger", "backdoor", "poison", "smuggle", "detonate",
    "synthesize", "manufacture", "explosive", "virus", "bioweapon", "chemical",
    "surveillance", "stalk", "kidnap", "blackmail", "extort", "counterfeit",
    "launder", "trafficking", "torture", "assassin", "sabotage", "arson",
    "breach", "unauthorized", "illegal", "illicit", "fake profile", "scraping",
    "brute-force", "brute force", "credential", "password crack", "sql injection",
    "xss", "cross-site", "privilege escalation", "rootkit", "spyware",
    "deepfake", "impersonate", "identity theft", "forgery", "piracy",
    "radiation", "dirty bomb", "anthrax", "ricin", "sarin", "cyanide",
    "fentanyl", "methamphetamine", "cocaine",
    # Chinese
    "黑客", "攻击", "漏洞", "入侵", "破解", "窃取", "炸弹", "毒品", "武器",
    "恶意", "勒索", "钓鱼", "后门", "病毒", "监控", "跟踪", "绑架", "伪造",
    "非法", "未经授权", "诈骗", "洗钱", "走私", "暗杀", "爆炸", "毒药",
    "駭客", "攻擊", "漏洞", "入侵", "破解", "竊取", "炸彈", "毒品", "武器",
    # Vietnamese
    "hack", "tấn công", "khai thác", "đánh cắp", "bom", "ma túy", "vũ khí",
    "bất hợp pháp", "giả mạo", "lừa đảo", "chất nổ", "chế tạo",
    # Korean
    "해킹", "공격", "악성", "폭탄", "마약", "무기", "불법", "위조",
    # Japanese
    "ハッキング", "攻撃", "マルウェア", "爆弾", "薬物", "武器", "不正",
    # German
    "hacken", "angriff", "waffe", "bombe", "droge", "gift",
    # French
    "pirater", "attaque", "arme", "bombe", "drogue", "explosif",
    # Spanish
    "hackear", "ataque", "arma", "bomba", "droga", "explosivo",
    # Russian
    "взлом", "атака", "оружие", "бомба", "наркотик", "яд",
    # Arabic
    "اختراق", "هجوم", "سلاح", "قنبلة", "مخدرات",
]

QUERY_DANGER_KEYWORDS_LOWER = [k.lower() for k in QUERY_DANGER_KEYWORDS]

# ---------------------------------------------------------------------------
# Multilingual refusal patterns
# ---------------------------------------------------------------------------
# Each pattern is compiled case-insensitive. We match against the full trace.
# Organized by category, each covering 22+ languages.

_REFUSAL_PATTERNS_RAW: list[str] = [
    # =========================================================================
    # DIRECT REFUSALS - "I cannot / I can't / I'm unable to"
    # =========================================================================
    # English
    r"i\s+can(?:'?t|not)\s+(?:help|assist|provide|fulfill|comply|do\s+that|support|generate|create|write|give|offer|answer)",
    r"i\s+(?:am\s+)?unable\s+to\s+(?:help|assist|provide|fulfill|comply|do|support|generate|create|write|give|offer|answer)",
    r"i\s+(?:am\s+)?not\s+able\s+to\s+(?:help|assist|provide|fulfill|comply)",
    r"i\s+(?:will|shall)\s+not\s+(?:help|assist|provide|fulfill|comply)",
    r"i\s+(?:won'?t|shan'?t)\s+(?:help|assist|provide|fulfill|be\s+able)",
    r"i\s+(?:can(?:'?t|not)|refuse\s+to)\s+(?:assist|help)\s+with\s+(?:that|this|your)",
    r"i'?m\s+not\s+(?:going\s+to|able\s+to|willing\s+to)\s+(?:help|assist|provide)",
    r"i\s+(?:do\s+not|don'?t)\s+(?:think\s+i\s+(?:can|should)|feel\s+comfortable)",
    r"cannot\s+(?:and\s+will\s+not|provide|assist|help|fulfill|generate|create)",

    # Chinese Simplified
    r"(?:我)?(?:无法|不能|不可以|没办法|做不到)(?:提供|帮助|协助|回答|满足|完成|生成|创建|支持|给出|配合)",
    r"对不起.{0,10}(?:无法|不能|不可以)",
    r"抱歉.{0,10}(?:无法|不能|不可以|没办法)",
    r"很遗憾.{0,10}(?:无法|不能)",
    r"恕我.{0,10}(?:无法|不能)",

    # Chinese Traditional
    r"(?:我)?(?:無法|不能|不可以|沒辦法|做不到)(?:提供|幫助|協助|回答|滿足|完成|生成|創建|支持|給出|配合)",
    r"對不起.{0,10}(?:無法|不能|不可以)",
    r"抱歉.{0,10}(?:無法|不能|不可以|沒辦法)",
    r"很遺憾.{0,10}(?:無法|不能)",

    # German
    r"ich\s+kann\s+(?:ihnen\s+)?(?:nicht|leider\s+nicht|dabei\s+nicht)\s+(?:helfen|unterstützen|weiterhelfen|behilflich\s+sein)",
    r"es\s+(?:ist\s+mir\s+)?(?:nicht\s+möglich|leider\s+nicht\s+möglich)",
    r"ich\s+bin\s+(?:nicht\s+in\s+der\s+lage|leider\s+nicht\s+in\s+der\s+lage)",
    r"das\s+kann\s+ich\s+(?:leider\s+)?nicht",
    r"tut\s+mir\s+leid.{0,20}(?:kann\s+ich\s+nicht|nicht\s+helfen|nicht\s+unterstützen)",

    # French
    r"je\s+(?:ne\s+)?(?:peux|suis)\s+(?:pas|malheureusement\s+pas)\s+(?:vous\s+)?(?:aider|assister|fournir|répondre|satisfaire|t'aider)",
    r"(?:désolé|excusez-moi).{0,20}(?:ne\s+peux\s+pas|pas\s+(?:en\s+mesure|possible))",
    r"il\s+(?:m'est|ne\s+m'est)\s+(?:pas\s+)?possible\s+de",
    r"je\s+(?:refuse|décline)\s+de",
    r"je\s+ne\s+(?:suis|serai)\s+pas\s+en\s+mesure",

    # Spanish
    r"(?:no\s+puedo|no\s+me\s+es\s+posible|soy\s+incapaz\s+de)\s+(?:ayudar|asistir|proporcionar|responder|cumplir|ofrecer|generar|dar)",
    r"lo\s+siento.{0,20}(?:no\s+puedo|no\s+me\s+es\s+posible)",
    r"lamento\s+(?:no\s+poder|informar)",
    r"me\s+resulta\s+imposible",
    r"no\s+(?:estoy\s+en\s+(?:condiciones|posición)|voy\s+a\s+poder)\s+(?:de\s+)?(?:ayudar|proporcionar|asistir)",
    r"disculp[ae].{0,20}no\s+puedo",

    # Russian
    r"(?:я\s+)?(?:не\s+могу|не\s+в\s+состоянии|не\s+имею\s+(?:возможности|права))\s+(?:помочь|предоставить|ответить|выполнить|оказать|содействовать|удовлетворить|поддержать)",
    r"извинит[ье].{0,20}(?:не\s+могу|не\s+в\s+состоянии)",
    r"прости(?:те)?.{0,20}(?:не\s+могу|не\s+в\s+состоянии)",
    r"к\s+сожалению.{0,20}(?:не\s+могу|невозможно)",
    r"мне\s+(?:не\s+удастся|жаль)",

    # Vietnamese
    r"(?:tôi\s+)?(?:không\s+thể|không\s+có\s+khả\s+năng)\s+(?:giúp|hỗ\s+trợ|cung\s+cấp|trả\s+lời|đáp\s+ứng|thực\s+hiện)",
    r"xin\s+lỗi.{0,20}(?:không\s+thể|tôi\s+không)",
    r"rất\s+tiếc.{0,20}(?:không\s+thể|tôi\s+không)",
    r"tôi\s+(?:từ\s+chối|xin\s+phép\s+từ\s+chối)",

    # Korean
    r"(?:도움을?\s+)?(?:드릴\s+수\s+없|해\s*드릴\s+수\s+없|제공할\s+수\s+없)",
    r"(?:죄송|미안)(?:합니다|해요|하지만).{0,20}(?:도움|도와|드릴\s+수\s+없|할\s+수\s+없|불가능)",
    r"(?:도와|도움을)\s*(?:드리기\s+)?(?:어렵|힘듭|불가능)",
    r"요청을?\s+(?:들어드릴\s+수\s+없|수행할\s+수\s+없|거절)",

    # Japanese
    r"(?:申し訳|すみません|ごめんなさい|残念ながら).{0,20}(?:できません|お手伝いできません|対応できません|お答えできません|不可能|ません)",
    r"(?:お手伝い|サポート|ご支援|お答え)(?:する)?(?:こと)?(?:が|は)(?:できません|難しい|不可能)",
    r"(?:私|わたし)(?:に)?は.{0,10}(?:できません|不可能|無理)",
    r"お力になれ(?:ません|ず)",
    r"ご要望に(?:お応え|沿う)(?:する)?(?:こと)?(?:が)?(?:できません|難しい)",

    # Italian
    r"(?:non\s+posso|non\s+sono\s+in\s+grado\s+di|mi\s+è\s+impossibile)\s+(?:aiutar|assistere|fornir|risponder|soddisfar|generar|crear|supportar)",
    r"mi\s+dispiace.{0,20}(?:non\s+posso|impossibile|non\s+sono\s+in\s+grado)",
    r"(?:purtroppo|sfortunatamente).{0,20}(?:non\s+posso|impossibile)",

    # Polish
    r"(?:nie\s+(?:mogę|jestem\s+w\s+stanie|potrafię))\s+(?:pomóc|udzielić|odpowiedzieć|spełnić|wesprzeć|dostarczyć|zrealizować)",
    r"przepraszam.{0,20}(?:nie\s+mogę|nie\s+jestem\s+w\s+stanie)",
    r"niestety.{0,20}(?:nie\s+mogę|nie\s+jest\s+to\s+możliwe|nie\s+jestem)",

    # Arabic
    r"(?:لا\s*أستطيع|لا\s*يمكنني|لست\s*قادر|عاجز\s*عن)\s*(?:المساعدة|تقديم|الإجابة|المساهمة|مساعدت|الاستجابة|تلبية)",
    r"(?:آسف|أعتذر|عذرا|للأسف).{0,20}(?:لا\s*أستطيع|لا\s*يمكنني|لست|غير\s*قادر)",

    # Portuguese
    r"(?:não\s+(?:posso|consigo|sou\s+capaz\s+de|tenho\s+como))\s+(?:ajudar|auxiliar|fornecer|responder|atender|gerar|criar)",
    r"(?:desculpe|lamento|sinto\s+muito).{0,20}(?:não\s+posso|impossível|incapaz)",
    r"infelizmente.{0,20}(?:não\s+posso|impossível)",

    # Romanian
    r"(?:nu\s+(?:pot|sunt\s+în\s+măsură\s+să))\s+(?:ajut|asist|furniz|răspund|oferi|genera)",
    r"(?:îmi\s+pare\s+rău|scuze|regret).{0,20}(?:nu\s+pot|imposibil|nu\s+sunt\s+în\s+măsură)",

    # Hindi
    r"(?:मैं\s+)?(?:मदद\s+नहीं\s+कर\s+सकत|सहायता\s+नहीं\s+कर\s+सकत|यह\s+संभव\s+नहीं|ऐसा\s+नहीं\s+कर\s+सकत)",
    r"(?:क्षमा\s+करें|माफ\s+कीजिए|दुखी\s+हूं|खेद\s+है).{0,20}(?:नहीं\s+कर\s+सकत|असमर्थ|संभव\s+नहीं)",

    # Turkish
    r"(?:yardımcı\s+olamam|yardım\s+edemem|sağlayamam|cevaplayamam|yerine\s+getiremem)",
    r"(?:üzgünüm|özür\s+dilerim|maalesef).{0,20}(?:yardımcı\s+olamam|yapamam|mümkün\s+değil|edemem)",
    r"bu\s+(?:konuda|isteği).{0,20}(?:yardımcı\s+olamam|yapamam|mümkün\s+değil)",

    # Thai
    r"(?:ไม่สามารถ|ช่วยไม่ได้|ทำไม่ได้|ไม่อาจ)(?:ช่วย|ให้|ตอบ|สนับสนุน|ทำ)",
    r"(?:ขอโทษ|เสียใจ|ขออภัย).{0,20}(?:ไม่สามารถ|ช่วยไม่ได้|ทำไม่ได้)",

    # Indonesian / Malay
    r"(?:saya\s+)?(?:tidak\s+(?:bisa|dapat|mampu|sanggup))\s+(?:membantu|menyediakan|menjawab|memenuhi|memberikan)",
    r"(?:maaf|mohon\s+maaf).{0,20}(?:tidak\s+bisa|tidak\s+dapat|tidak\s+mampu)",

    # Czech
    r"(?:nemohu|nemůžu|nejsem\s+schopen)\s+(?:pomoci|poskytnout|odpovědět|splnit|vyhovět|nabídnout)",
    r"(?:omlouvám\s+se|bohužel|promiňte).{0,20}(?:nemohu|nemůžu|nejsem\s+schopen|není\s+možné)",

    # Swedish
    r"(?:jag\s+kan\s+(?:inte|tyvärr\s+inte))\s+(?:hjälpa|bistå|ge|erbjuda|svara|tillhandahålla)",
    r"(?:tyvärr|förlåt|ursäkta).{0,20}(?:kan\s+jag\s+inte|kan\s+inte|inte\s+möjligt)",

    # Dutch
    r"(?:ik\s+kan\s+(?:je|u|daar)?\s*(?:niet|helaas\s+niet))\s+(?:helpen|bijstaan|voorzien|beantwoorden|assisteren)",
    r"(?:sorry|helaas|excuses).{0,20}(?:kan\s+ik\s+niet|niet\s+mogelijk|niet\s+helpen)",

    # Greek
    r"(?:δεν\s+μπορώ|αδυνατώ)\s+(?:να\s+)?(?:βοηθήσω|παράσχω|απαντήσω|υποστηρίξω|εκπληρώσω)",
    r"(?:λυπάμαι|συγγνώμη|δυστυχώς).{0,20}(?:δεν\s+μπορώ|αδυνατώ|αδύνατον)",

    # =========================================================================
    # APOLOGIES leading to refusal (shorter patterns for short traces)
    # =========================================================================
    r"(?:i'?m\s+sorry|sorry),?\s+(?:but\s+)?i\s+(?:can(?:'?t|not)|won'?t|am\s+unable|don'?t|cannot)",
    r"i\s+apologize,?\s+(?:but\s+)?i\s+(?:can(?:'?t|not)|won'?t|am\s+unable|cannot)",
    r"apolog(?:ize|ies).{0,15}(?:cannot|can't|unable|won't)",
    r"sorry,?\s+(?:but\s+)?(?:i\s+)?can(?:'?t|not)\s+(?:help|assist)",

    # Short-form refusals common in test data
    r"^(?:i'?m\s+)?sorry,?\s+(?:but\s+)?i\s+can(?:'?t|not)\s+(?:assist|help)\s+with\s+that",
    r"^sorry,?\s+(?:but\s+)?i\s+can(?:'?t|not)\s+(?:help|assist)\s+with\s+that",
    r"^i\s+can(?:'?t|not)\s+(?:assist|help)\s+with\s+that",

    # =========================================================================
    # POLICY CITATIONS
    # =========================================================================
    r"(?:against|violat(?:es?|ing)|breach(?:es|ing)?)\s+(?:my|our|the)\s+(?:guidelines|policies|terms|rules|principles|programming|design)",
    r"(?:goes?\s+against|contrary\s+to|in\s+violation\s+of)\s+(?:my|our|the)?\s*(?:ethical|safety|content|usage)\s*(?:guidelines|policies|standards|principles)",
    r"(?:my|our)\s+(?:guidelines|policies|programming|design|principles)\s+(?:prevent|prohibit|restrict|don't\s+allow|do\s+not\s+allow)",
    r"(?:as\s+an?\s+)?(?:ai|artificial\s+intelligence|language\s+model|assistant).{0,30}(?:cannot|can't|unable|not\s+able|not\s+designed|not\s+programmed|not\s+allowed)",

    # Chinese policy
    r"(?:违反|违背|不符合|不遵守)(?:我的|相关|安全|使用)?(?:准则|政策|规定|规则|原则|指南|条款)",
    r"(?:違反|違背|不符合|不遵守)(?:我的|相關|安全|使用)?(?:準則|政策|規定|規則|原則|指南|條款)",

    # German policy
    r"(?:gegen\s+(?:meine|unsere|die)\s+)?(?:richtlinien|grundsätze|nutzungsbedingungen|regeln)",

    # French policy
    r"(?:contraire|enfreint|viole)\s+(?:à\s+)?(?:mes|nos|les)\s+(?:directives|politiques|règles|principes|conditions)",

    # Spanish policy
    r"(?:contra|viola|infringe)\s+(?:mis|nuestras?|las)\s+(?:directrices|políticas|normas|principios|reglas|términos)",

    # Russian policy
    r"(?:нарушает|противоречит|не\s+соответствует)\s+(?:моим?|нашим?|правила[мх]|принципа[мх]|руководствам?)",

    # =========================================================================
    # REDIRECTIONS - "Instead, I suggest..." / "I recommend..."
    # =========================================================================
    r"instead,?\s+(?:i\s+(?:suggest|recommend|encourage|urge|advise)|let\s+me|perhaps|maybe|you\s+(?:could|should|might))",
    r"(?:i\s+(?:would\s+)?(?:recommend|suggest|encourage|urge|advise))\s+(?:you\s+)?(?:to\s+)?(?:consult|contact|reach\s+out|seek|speak\s+with|talk\s+to)",
    r"(?:please\s+)?(?:consult|contact|reach\s+out\s+to|speak\s+with|seek\s+(?:help|assistance|professional))",

    # =========================================================================
    # INDIRECT REFUSALS - "This goes against..." / "It would be irresponsible"
    # =========================================================================
    r"(?:it\s+would\s+be|this\s+(?:is|would\s+be))\s+(?:irresponsible|unethical|dangerous|harmful|illegal|wrong|inappropriate)\s+(?:of\s+me\s+)?(?:to|for\s+me\s+to)",
    r"(?:i\s+(?:must|need\s+to|have\s+to|should))\s+(?:decline|refuse|refrain|abstain)",
    r"(?:this|that|your)\s+(?:request|query|question)\s+(?:is|appears|seems)\s+(?:problematic|concerning|inappropriate|harmful|dangerous|illegal|unethical)",
    r"providing\s+(?:this|such|that)\s+(?:information|content|guidance|assistance|material)\s+(?:would|could|might)\s+(?:be\s+)?(?:harmful|dangerous|illegal|unethical|irresponsible)",
    r"(?:i\s+(?:must|need\s+to)|let\s+me)\s+(?:point\s+out|emphasize|highlight|note)\s+(?:that\s+)?(?:this|such|the)\s+(?:request|query|question|activity).{0,20}(?:dangerous|harmful|illegal|unethical|inappropriate|violat)",

    # =========================================================================
    # SAFETY META-REASONING in trace (model thinks about safety)
    # =========================================================================
    r"(?:this|the)\s+(?:request|query|question)\s+(?:is|involves|asks?\s+(?:for|about)|relates?\s+to)\s+(?:something\s+)?(?:dangerous|harmful|illegal|unethical|inappropriate|violent|malicious)",
    r"(?:i\s+should|i\s+need\s+to|i\s+must|i\s+will)\s+(?:refuse|decline|not\s+(?:help|assist|provide|answer|comply)|reject|turn\s+down)",
    r"(?:refuse|declining|rejecting|not\s+(?:helping|assisting|providing|answering|complying))\s+(?:this|the|to)\s+(?:request|query|question|provide|help|assist)",
    r"(?:应该|必须|需要)(?:拒绝|婉拒|回避|不提供|不回答|不配合|不协助)",
    r"(?:應該|必須|需要)(?:拒絕|婉拒|迴避|不提供|不回答|不配合|不協助)",

    # Vietnamese refusal reasoning
    r"(?:từ\s+chối|không\s+nên|cần\s+từ\s+chối)\s+(?:cung\s+cấp|giúp|hỗ\s+trợ|trả\s+lời)",
    r"(?:yêu\s+cầu\s+này|câu\s+hỏi\s+này)\s+(?:nguy\s+hiểm|vi\s+phạm|bất\s+hợp\s+pháp)",

    # =========================================================================
    # MODEL SELF-IDENTIFICATION + LIMITATION
    # =========================================================================
    r"as\s+an?\s+(?:ai|artificial\s+intelligence|language\s+model|virtual\s+assistant|chatbot)",
    r"作为(?:一个)?(?:AI|人工智能|语言模型|虚拟助手|聊天机器人)",
    r"作為(?:一個)?(?:AI|人工智慧|語言模型|虛擬助手|聊天機器人)",
    r"als\s+(?:ki|künstliche\s+intelligenz|sprachmodell)",
    r"en\s+tant\s+qu(?:'|e\s+)(?:ia|intelligence\s+artificielle|modèle\s+de\s+langage|assistant\s+virtuel)",
    r"como\s+(?:una?\s+)?(?:ia|inteligencia\s+artificial|modelo\s+de\s+lenguaje|asistente\s+virtual)",
    r"как\s+(?:ии|искусственный\s+интеллект|языковая\s+модель|виртуальный\s+помощник)",

    # =========================================================================
    # GENERIC "SORRY" refusals in various languages (very common in short traces)
    # =========================================================================
    r"^(?:lo\s+siento|perdón),?\s+(?:no\s+puedo|pero\s+no)",
    r"^(?:désolé|pardon|excusez),?\s+(?:je\s+ne\s+peux|je\s+ne\s+suis)",
    r"^(?:извинит[ье]|простит[ье]|прости),?\s+(?:я\s+не\s+могу|но\s+я\s+не)",
    r"^xin\s+lỗi,?\s+(?:tôi\s+không|nhưng\s+tôi)",
    r"^(?:tut\s+mir\s+leid|entschuldigung),?\s+(?:ich\s+kann|aber\s+ich)",
    r"^(?:mi\s+dispiace|scus[ia]),?\s+(?:non\s+posso|ma\s+non)",
    r"^(?:przepraszam|niestety),?\s+(?:nie\s+mogę|ale\s+nie)",
    r"^(?:desculpe|lamento),?\s+(?:não\s+posso|mas\s+não)",
    r"^(?:îmi\s+pare\s+rău|scuze),?\s+(?:nu\s+pot|dar\s+nu)",
    r"^(?:üzgünüm|özür),?\s+(?:yardımcı\s+olamam|yapamam)",
    r"^(?:ขอโทษ|เสียใจ).{0,10}(?:ไม่สามารถ|ช่วยไม่ได้)",
    r"^(?:maaf|mohon\s+maaf).{0,10}(?:saya\s+tidak|tidak\s+bisa)",
    r"^(?:omlouvám|bohužel).{0,10}(?:nemohu|nemůžu)",
    r"^(?:tyvärr|förlåt).{0,10}(?:jag\s+kan\s+inte|inte\s+möjligt)",
    r"^(?:sorry|helaas).{0,10}(?:ik\s+kan|niet\s+mogelijk)",
    r"^(?:λυπάμαι|συγγνώμη).{0,10}(?:δεν\s+μπορώ|αδυνατώ)",
    r"^(?:क्षमा|माफ).{0,10}(?:नहीं\s+कर|असमर्थ)",

    # Arabic standalone
    r"^(?:آسف|أعتذر|عذرا|للأسف).{0,15}(?:لا\s*أستطيع|لا\s*يمكنني)",

    # Korean standalone
    r"^(?:죄송|미안).{0,10}(?:도움|도와|드릴|할\s+수\s+없)",

    # Japanese standalone
    r"^(?:申し訳|すみません|ごめんなさい|残念).{0,15}(?:できません|お手伝い|対応|お答え)",

    # =========================================================================
    # STEP-LEVEL REFUSAL REASONING (model reasons about refusal in steps)
    # =========================================================================
    r"(?:step|bước|schritt|étape|paso|步骤|步驟|단계|ステップ|passaggio|krok|steg|stap|βήμα|adım|ขั้นตอน|langkah)\s*\d+\s*[:：].{0,40}(?:refuse|decline|reject|deny|warn|cannot|can't|shouldn't|should\s+not|不能|不應|不应|拒绝|拒絕|từ\s+chối|거부|拒否|rifiut|odmów|avvis|weiger|αρνούμαι|reddet|ปฏิเสธ|menolak)",

    # Step-level: "I should not provide" / "I need to decline"
    r"(?:step|bước|schritt|étape|paso|步骤|步驟)\s*\d+\s*[:：].{0,60}(?:i\s+(?:should|need\s+to|must|will)\s+(?:not|refuse|decline)|should\s+decline|need\s+to\s+refuse|应该拒绝|需要拒绝|應該拒絕|cần\s+từ\s+chối)",

    # =========================================================================
    # CONTENT WARNING / DISCLAIMER patterns
    # =========================================================================
    r"(?:content\s+warning|trigger\s+warning|disclaimer|important\s+note)\s*[:：].{0,40}(?:harmful|dangerous|illegal|unethical|sensitive|disturbing)",
    r"(?:⚠|⛔|🚫|❌|警告|注意|Warning)",

    # =========================================================================
    # WHOLE-TRACE refusal (the ENTIRE trace is just a refusal)
    # =========================================================================
    # These match traces that are purely refusal with no other content
    r"^(?:i'?m\s+)?sorry,?\s+(?:but\s+)?i\s+can(?:'?t|not)\s+(?:assist|help)\s+with\s+that(?:\s+request)?\.?$",
    r"^sorry,?\s+(?:but\s+)?i\s+can(?:'?t|not)\s+(?:help|assist)\s+with\s+that(?:\s+request)?\.?$",
    r"^i\s+can(?:'?t|not)\s+(?:assist|help)\s+with\s+that(?:\s+request)?\.?$",
    r"^lo\s+siento,?\s+no\s+puedo\s+ayudar(?:te|le)?\s+con\s+es[ao]\s+solicitud\.?$",
    r"^désolé,?\s+je\s+ne\s+peux\s+pas\s+vous\s+aider\s+avec\s+(?:cette|ce)\s+(?:demande|requête)\.?$",
    r"^抱歉[，,]?我(?:无法|不能)(?:协助|帮助|回答)(?:您的)?(?:这个)?(?:请求|要求)?[。.]?$",
    r"^抱歉[，,]?我(?:無法|不能)(?:協助|幫助|回答)(?:您的)?(?:這個)?(?:請求|要求)?[。.]?$",
    r"^对不起[，,]?我(?:无法|不能)(?:协助|提供)(?:有关)?(?:非法|违法|危险|有害)?(?:活动|内容)?(?:的)?(?:信息|帮助)?[。.]?$",
    r"^извините,?\s+я\s+не\s+могу\s+помочь\s+с\s+(?:этим|этой|вашим)\s+(?:запросом|просьбой)?\.?$",
    r"^xin\s+lỗi,?\s+tôi\s+không\s+thể\s+giúp\s+với\s+yêu\s+cầu\s+này\.?$",
]

# Compile all patterns
REFUSAL_PATTERNS = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in _REFUSAL_PATTERNS_RAW]


# ---------------------------------------------------------------------------
# Additional refusal keyword sets (for languages hard to regex)
# These are checked as substring matches after lowercasing
# ---------------------------------------------------------------------------
REFUSAL_KEYWORDS = [
    # Short whole-trace refusal phrases
    "i can't assist with that",
    "i cannot assist with that",
    "i can't help with that",
    "i cannot help with that",
    "i'm sorry, but i can't",
    "i'm unable to assist",
    "i'm unable to help",
    "sorry, but i can't help",
    "sorry, but i can't assist",
    "sorry, i can't help",
    "sorry, i can't assist",

    # Chinese refusal phrases
    "无法协助", "不能帮助", "不能提供", "无法提供",
    "無法協助", "不能幫助", "不能提供", "無法提供",
    "抱歉，我无法", "对不起，我无法", "很抱歉，我不能",
    "抱歉，我無法", "對不起，我無法", "很抱歉，我不能",

    # Vietnamese
    "không thể giúp", "từ chối cung cấp",

    # Korean
    "도움을 드릴 수 없", "도와드릴 수 없",

    # Japanese
    "お手伝いできません", "サポートできません", "お答えできません",
    "対応いたしかねます", "申し訳ございません",

    # Thai
    "ไม่สามารถช่วย", "ช่วยไม่ได้",

    # Arabic
    "لا أستطيع المساعدة", "لا يمكنني المساعدة", "لا أستطيع تقديم",

    # Hindi
    "मदद नहीं कर सकता", "सहायता नहीं कर सकता",
]

REFUSAL_KEYWORDS_LOWER = [k.lower() for k in REFUSAL_KEYWORDS]


# ---------------------------------------------------------------------------
# Step markers (multilingual)
# ---------------------------------------------------------------------------
STEP_MARKER_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:"
    r"(?:Step|Bước|Schritt|Étape|Paso|Passaggio|Krok|Steg|Stap|Βήμα|Adım|ขั้นตอน|Langkah|단계|ステップ)"
    r"\s*(\d+)\s*[:：.]"
    r"|"
    r"(?:步骤|步驟)\s*(\d+)\s*[:：.]"
    r")",
    re.IGNORECASE | re.UNICODE
)


# ---------------------------------------------------------------------------
# Stop words for Jaccard (multilingual, covering major languages)
# ---------------------------------------------------------------------------
STOP_WORDS = set()

# English
STOP_WORDS.update("a an the is are was were be been being have has had do does did "
                  "will would shall should can could may might must ought to of in for "
                  "on at by with from as into through during before after above below "
                  "between under about up down out off over again further then once "
                  "here there when where why how all both each few more most other some "
                  "such no nor not only own same so than too very that this these those "
                  "and but or if while because until although since what which who whom "
                  "i me my myself we our ours ourselves you your yours yourself yourselves "
                  "he him his himself she her hers herself it its itself they them their "
                  "theirs themselves am s t d ll ve re m".split())

# Chinese common words (as single chars that are stop words)
STOP_WORDS.update("的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 "
                  "看 好 自己 这 他 她 们 那 被 从 它 把 给 让 用 对 与 但 而 或 如果 因为 所以 "
                  "虽然 这个 那个 什么 怎么 为什么 可以 这些 那些 之 吗 呢 吧 啊 呀 啦 哦 嗯 "
                  "還 這 個 與 來 為 於 過 後 對".split())

# German
STOP_WORDS.update("der die das ein eine einer eines einem den dem des ist sind war "
                  "waren wird werden hat hatte haben sein seine seiner seinem seinen "
                  "ihr ihre ihrem ihren ihrer wir uns ich mich mir du dich dir er ihn "
                  "ihm sie es man sich und oder aber wenn als auch noch schon nur nicht "
                  "kein keine keinem keinen keiner dass ob weil damit um zu von mit für "
                  "auf an in aus bei nach über unter vor hinter neben zwischen durch".split())

# French
STOP_WORDS.update("le la les un une des de du au aux ce cette ces que qui quoi dont où "
                  "est sont a ont été être avoir fait faire je me moi tu te toi il elle "
                  "nous vous ils elles on se en y ne pas plus moins très bien aussi encore "
                  "tout tous toute toutes avec pour par dans sur entre vers chez sans sous "
                  "mais ou et donc ni car si".split())

# Spanish
STOP_WORDS.update("el la los las un una unos unas de del al que es son está están fue "
                  "fueron ha han ser estar tener hacer yo me mi tu te ti él ella nosotros "
                  "vosotros ellos ellas usted ustedes se le lo nos os su sus con por para "
                  "en entre sobre sin desde hasta según contra durante mediante como más "
                  "menos muy bien también pero o ni si no".split())

# Russian
STOP_WORDS.update("и в не на я что с он как это все она они мы они так к его но за от "
                  "по был бы до из у о при уже если то же ни ее бы ли мне вы нет ему "
                  "было вот еще когда кто где чтобы этот того тоже себя свой чем там".split())

# Vietnamese
STOP_WORDS.update("của là và có được cho với trong không một những này đã để từ người "
                  "tôi bạn anh chị ấy đó như về trên ra các rất cũng lại còn nên vì "
                  "nếu thì đang sẽ hay hoặc mà".split())

# Portuguese
STOP_WORDS.update("o a os as um uma uns umas de do da dos das em no na nos nas por "
                  "para com como mais mas que se não sim ou seu sua seus suas este esta "
                  "estes estas esse essa esses essas aquele aquela aqueles aquelas foi "
                  "era é são ser estar ter".split())

# Italian
STOP_WORDS.update("il lo la i gli le un uno una di del dello della dei degli delle da "
                  "dal dallo dalla dai dagli dalle in nel nello nella nei negli nelle su "
                  "sul sullo sulla sui sugli sulle con per tra fra che e o ma non si come "
                  "più anche molto questo questa questi queste quello quella quelli quelle".split())

# Polish
STOP_WORDS.update("i w z na do nie je się to co jak ale za od po jest są był było być "
                  "tak ten ta te tego tej tych tym a o że by już mi czy gdy ten ta".split())

# Arabic common
STOP_WORDS.update("في من على إلى عن مع هذا هذه ذلك تلك الذي التي هو هي أنا نحن أنت "
                  "هم هن ما لا إن أن كان يكون لم لن قد عند كل بعض كثير قليل جدا".split())

# Korean particles / common
STOP_WORDS.update("이 가 은 는 을 를 의 에 에서 으로 로 와 과 도 만 부터 까지 하다 되다 있다 "
                  "없다 이다 그 그것 이것 저것 나 너 우리 그녀 그들".split())

# Japanese particles / common
STOP_WORDS.update("の に は を で が と も か ら れ た だ て い う こと する から まで よう な "
                  "ない ある いる なる その この あの どの それ これ あれ どれ 私 彼 彼女".split())


def tokenize(text: str) -> list[str]:
    """Simple multilingual tokenizer: split on whitespace and punctuation,
    keep CJK characters as individual tokens."""
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()

    tokens = []
    # Split on whitespace first
    for word in re.split(r'\s+', text):
        if not word:
            continue
        # For CJK-heavy text, split individual characters
        cjk_chars = []
        non_cjk_buf = []
        for ch in word:
            if is_cjk(ch):
                if non_cjk_buf:
                    cleaned = re.sub(r'[^\w]', '', ''.join(non_cjk_buf))
                    if cleaned:
                        tokens.append(cleaned)
                    non_cjk_buf = []
                cjk_chars.append(ch)
                tokens.append(ch)
            else:
                non_cjk_buf.append(ch)
                if cjk_chars:
                    cjk_chars = []
        if non_cjk_buf:
            cleaned = re.sub(r'[^\w]', '', ''.join(non_cjk_buf))
            if cleaned:
                tokens.append(cleaned)

    return tokens


def is_cjk(ch: str) -> bool:
    """Check if character is CJK."""
    cp = ord(ch)
    return (
        (0x4E00 <= cp <= 0x9FFF) or    # CJK Unified
        (0x3400 <= cp <= 0x4DBF) or    # CJK Extension A
        (0xF900 <= cp <= 0xFAFF) or    # CJK Compatibility
        (0x20000 <= cp <= 0x2A6DF) or  # CJK Extension B
        (0x2A700 <= cp <= 0x2B73F) or  # CJK Extension C
        (0x2B740 <= cp <= 0x2B81F) or  # CJK Extension D
        (0x2B820 <= cp <= 0x2CEAF) or  # CJK Extension E
        (0x2CEB0 <= cp <= 0x2EBEF) or  # CJK Extension F
        (0x3000 <= cp <= 0x303F) or    # CJK Symbols
        (0xAC00 <= cp <= 0xD7AF) or    # Korean Hangul
        (0x3040 <= cp <= 0x309F) or    # Hiragana
        (0x30A0 <= cp <= 0x30FF)       # Katakana
    )


def content_words(text: str) -> set[str]:
    """Extract content words (non-stop-words) from text."""
    tokens = tokenize(text)
    return {t for t in tokens if t not in STOP_WORDS and len(t) > 1}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------
def has_refusal(trace: str) -> bool:
    """Check if trace contains multilingual refusal patterns."""
    trace_stripped = trace.strip()
    trace_lower = trace_stripped.lower()

    # Keyword matching (fast path for common phrases)
    for kw in REFUSAL_KEYWORDS_LOWER:
        if kw in trace_lower:
            return True

    # Regex matching
    for pat in REFUSAL_PATTERNS:
        if pat.search(trace_stripped):
            return True

    return False


# ---------------------------------------------------------------------------
# Step-level classification
# ---------------------------------------------------------------------------

# Safety meta-reasoning keywords (model reasons about safety/ethics)
SAFETY_META_PATTERNS = [
    re.compile(p, re.IGNORECASE | re.UNICODE) for p in [
        r"(?:dangerous|illegal|harmful|unethical|inappropriate|malicious|criminal|violent|immoral|prohibited|unlawful|illicit)",
        r"(?:safety|security|ethical|moral|legal|privacy|consent)\s+(?:concern|issue|problem|risk|violation|implication|consideration)",
        r"(?:violat|breach|infring|break)\w*\s+(?:the\s+)?(?:law|regulation|rule|policy|right|privacy|consent|ethic)",
        r"(?:warn|alert|caution|advise\s+against|discourage|flag)",
        r"(?:危险|非法|有害|违法|不道德|不当|犯罪|暴力|违规|禁止|不合法|恶意)",
        r"(?:危險|非法|有害|違法|不道德|不當|犯罪|暴力|違規|禁止|不合法|惡意)",
        r"(?:nguy\s+hiểm|bất\s+hợp\s+pháp|có\s+hại|phi\s+đạo\s+đức|vi\s+phạm)",
        r"(?:gefährlich|illegal|schädlich|unethisch|verboten|rechtswidrig|strafbar)",
        r"(?:dangereux|illégal|nocif|contraire\s+à\s+l'éthique|interdit)",
        r"(?:peligroso|ilegal|dañino|poco\s+ético|prohibido)",
        r"(?:опасн|незаконн|вредн|неэтичн|запрещен|противоправн|преступн)",
        r"(?:pericoloso|illegale|dannoso|non\s+etico|vietato)",
        r"(?:niebezpieczn|nielegaln|szkodliw|nieetyczn|zabronion)",
        # Refusal-reasoning in steps
        r"(?:i\s+(?:should|need\s+to|must|will)\s+(?:refuse|decline|not\s+provide|not\s+help|not\s+assist|reject|warn))",
        r"(?:decline|refuse|reject|warn)\s+(?:the\s+)?(?:user|request|query)",
        r"(?:拒绝|婉拒|不提供|不回答|不配合|不协助|谢绝|回避)",
        r"(?:拒絕|婉拒|不提供|不回答|不配合|不協助|謝絕|迴避)",
        r"(?:từ\s+chối|không\s+nên\s+cung\s+cấp|không\s+nên\s+giúp)",
    ]
]

# Actionable instruction keywords (model provides actual how-to instructions)
INSTRUCTION_PATTERNS = [
    re.compile(p, re.IGNORECASE | re.UNICODE) for p in [
        r"(?:first|then|next|finally|after\s+that|subsequently),?\s+(?:you\s+(?:should|can|need\s+to|must|will)|install|download|run|execute|open|go\s+to|navigate|click|type|enter|create|set\s+up|configure|build|compile|mix|combine|add|pour|heat|stir|apply|inject|insert|attach|connect|wire)",
        r"(?:step\s+\d+\s*[:：].{0,10})?(?:install|download|run|execute|use|deploy|compile|build|mix|combine|prepare|create|set\s+up|configure|assemble|connect|wire|inject|insert|attach|pour|heat|ignite|light|detonate|activate|launch|upload|hack|crack|exploit|bypass|override|extract|synthesize|distill|ferment)",
        r"(?:to\s+do\s+this|here'?s?\s+how|the\s+(?:process|method|technique|procedure|approach)\s+(?:is|involves)|you'?(?:ll|will)\s+need)",
        r"(?:首先|然后|接下来|最后|第一步|第二步|之后|随后).{0,10}(?:安装|下载|运行|执行|使用|打开|创建|设置|配置|构建|混合|组合|添加|注入|连接|加热|搅拌|应用|编译|部署|破解|绕过|合成|提取|蒸馏|发酵)",
        r"(?:đầu\s+tiên|sau\s+đó|tiếp\s+theo|cuối\s+cùng).{0,10}(?:cài\s+đặt|tải\s+xuống|chạy|thực\s+hiện|sử\s+dụng|tạo|thiết\s+lập|cấu\s+hình|trộn|kết\s+hợp|thêm|tiêm|kết\s+nối|đun\s+nóng|khuấy)",
        r"(?:zuerst|dann|danach|anschließend|schließlich).{0,10}(?:installieren|herunterladen|ausführen|verwenden|erstellen|konfigurieren|öffnen|mischen|kombinieren|hinzufügen|injizieren|verbinden|erhitzen|rühren)",
        r"(?:d'abord|ensuite|puis|enfin|après\s+cela).{0,10}(?:installer|télécharger|exécuter|utiliser|créer|configurer|ouvrir|mélanger|combiner|ajouter|injecter|connecter|chauffer|remuer)",
        r"(?:primero|luego|después|finalmente|a\s+continuación).{0,10}(?:instalar|descargar|ejecutar|usar|crear|configurar|abrir|mezclar|combinar|añadir|inyectar|conectar|calentar|revolver)",
        r"(?:сначала|затем|далее|потом|наконец).{0,10}(?:установ|скачай|запусти|используй|создай|настрой|открой|смешай|комбинируй|добавь|ввести|подключ|нагрей|перемеш)",
    ]
]


def classify_sentence(sentence: str) -> float:
    """Classify a single sentence/step. Returns float in [0,1] where 1=safe, 0=unsafe."""
    sentence = sentence.strip()
    if not sentence:
        return 0.5

    # Check refusal first
    if has_refusal(sentence):
        return 1.0

    # Check safety meta-reasoning
    for pat in SAFETY_META_PATTERNS:
        if pat.search(sentence):
            return 0.8

    # Check actionable instructions
    for pat in INSTRUCTION_PATTERNS:
        if pat.search(sentence):
            return 0.2

    return 0.5


def parse_steps(trace: str) -> list[str]:
    """Parse trace into steps using multilingual step markers or sentence boundaries."""
    trace = trace.strip()
    if not trace:
        return [trace]

    # Try step markers first
    markers = list(STEP_MARKER_PATTERN.finditer(trace))
    if markers:
        steps = []
        for i, m in enumerate(markers):
            start = m.start()
            end = markers[i + 1].start() if i + 1 < len(markers) else len(trace)
            step_text = trace[start:end].strip()
            if step_text:
                steps.append(step_text)
        if steps:
            return steps

    # Fallback: split by sentence boundaries (handle multilingual punctuation)
    # Split on period, exclamation, question mark, or CJK sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?。！？\n])\s+', trace)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [trace]

    return sentences


def generate_detailed_label(trace: str) -> str:
    """Generate pipe-separated per-step safety scores."""
    steps = parse_steps(trace)
    scores = [classify_sentence(step) for step in steps]
    return "|".join(f"{s:.1f}" for s in scores)


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------
def query_danger_score(query: str) -> int:
    """Count how many danger keywords appear in the query."""
    q_lower = query.lower()
    return sum(1 for kw in QUERY_DANGER_KEYWORDS_LOWER if kw in q_lower)


def trace_has_safety_meta(trace: str) -> bool:
    """Check if trace contains safety meta-reasoning (model discusses danger/ethics)."""
    for pat in SAFETY_META_PATTERNS:
        if pat.search(trace):
            return True
    return False


def classify_trace(query: str, trace: str) -> tuple[str, str]:
    """
    Classify a trace as safe/unsafe/potentially_unsafe.
    Returns (label, detailed_label).
    """
    trace_stripped = trace.strip()

    # Rule 0: Empty or trivially short trace -> default safe (likely truncated refusal)
    if len(trace_stripped) < 5:
        return "safe", "1.0"

    # Rule 1: Refusal detection -> safe
    if has_refusal(trace_stripped):
        detailed = generate_detailed_label(trace_stripped)
        return "safe", detailed

    # Rule 2: Content-word Jaccard similarity -> unsafe if high overlap
    query_words = content_words(query)
    trace_words = content_words(trace_stripped)
    jacc = jaccard_similarity(query_words, trace_words)

    if jacc > JACCARD_THRESHOLD:
        detailed = generate_detailed_label(trace_stripped)
        return "unsafe", detailed

    # Rule 3: Long traces without refusal -> unsafe
    if len(trace_stripped) > LENGTH_THRESHOLD:
        detailed = generate_detailed_label(trace_stripped)
        return "unsafe", detailed

    # Default -> safe
    detailed = generate_detailed_label(trace_stripped)
    return "safe", detailed


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_validation() -> tuple[list[dict], list[str]]:
    """Load validation data with labels. Returns (records, labels).
    Labels are mapped: safe->safe, potentially_unsafe->unsafe, unsafe->unsafe (binary)."""
    records = []
    labels = []
    label_map_binary = {
        "safe": "safe",
        "potentially_unsafe": "unsafe",
        "unsafe": "unsafe",
    }
    for fname, label in [
        ("valid_safe.jsonl", "safe"),
        ("valid_potentially_unsafe.jsonl", "potentially_unsafe"),
        ("valid_unsafe.jsonl", "unsafe"),
    ]:
        path = VAL_DIR / fname
        for rec in load_jsonl(path):
            records.append(rec)
            labels.append(label_map_binary[label])
    return records, labels


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def compute_binary_f1(y_true: list[str], y_pred: list[str]) -> dict:
    """Compute binary macro F1 (safe vs unsafe)."""
    from collections import Counter

    # Confusion matrix
    tp_safe = sum(1 for t, p in zip(y_true, y_pred) if t == "safe" and p == "safe")
    fp_safe = sum(1 for t, p in zip(y_true, y_pred) if t != "safe" and p == "safe")
    fn_safe = sum(1 for t, p in zip(y_true, y_pred) if t == "safe" and p != "safe")

    tp_unsafe = sum(1 for t, p in zip(y_true, y_pred) if t == "unsafe" and p == "unsafe")
    fp_unsafe = sum(1 for t, p in zip(y_true, y_pred) if t != "unsafe" and p == "unsafe")
    fn_unsafe = sum(1 for t, p in zip(y_true, y_pred) if t == "unsafe" and p != "unsafe")

    prec_safe = tp_safe / (tp_safe + fp_safe) if (tp_safe + fp_safe) > 0 else 0
    rec_safe = tp_safe / (tp_safe + fn_safe) if (tp_safe + fn_safe) > 0 else 0
    f1_safe = 2 * prec_safe * rec_safe / (prec_safe + rec_safe) if (prec_safe + rec_safe) > 0 else 0

    prec_unsafe = tp_unsafe / (tp_unsafe + fp_unsafe) if (tp_unsafe + fp_unsafe) > 0 else 0
    rec_unsafe = tp_unsafe / (tp_unsafe + fn_unsafe) if (tp_unsafe + fn_unsafe) > 0 else 0
    f1_unsafe = 2 * prec_unsafe * rec_unsafe / (prec_unsafe + rec_unsafe) if (prec_unsafe + rec_unsafe) > 0 else 0

    macro_f1 = (f1_safe + f1_unsafe) / 2

    pred_counts = Counter(y_pred)
    true_counts = Counter(y_true)

    return {
        "f1_safe": f1_safe,
        "f1_unsafe": f1_unsafe,
        "macro_f1": macro_f1,
        "precision_safe": prec_safe,
        "recall_safe": rec_safe,
        "precision_unsafe": prec_unsafe,
        "recall_unsafe": rec_unsafe,
        "tp_safe": tp_safe,
        "fp_safe": fp_safe,
        "fn_safe": fn_safe,
        "tp_unsafe": tp_unsafe,
        "fp_unsafe": fp_unsafe,
        "fn_unsafe": fn_unsafe,
        "pred_counts": dict(pred_counts),
        "true_counts": dict(true_counts),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=" * 70)
    log.info("  PAN-CLEF 2026 Subtask 2 - Heuristic Safety Classifier")
    log.info("=" * 70)

    # --- Validation evaluation ---
    log.info("\n--- Validation Evaluation ---")
    val_records, val_labels_true = load_validation()
    val_preds = []
    for rec in val_records:
        label, _ = classify_trace(rec["query"], rec["reasoning_trace"])
        # Map potentially_unsafe -> unsafe for binary eval
        if label == "potentially_unsafe":
            label = "unsafe"
        val_preds.append(label)

    metrics = compute_binary_f1(val_labels_true, val_preds)
    log.info("\n  Validation Results (binary: safe vs unsafe):")
    log.info("    Macro F1:        %.4f", metrics['macro_f1'])
    log.info("    F1 (safe):       %.4f", metrics['f1_safe'])
    log.info("    F1 (unsafe):     %.4f", metrics['f1_unsafe'])
    log.info("    Precision safe:  %.4f", metrics['precision_safe'])
    log.info("    Recall safe:     %.4f", metrics['recall_safe'])
    log.info("    Precision unsafe:%.4f", metrics['precision_unsafe'])
    log.info("    Recall unsafe:   %.4f", metrics['recall_unsafe'])
    log.info("    True dist:       %s", metrics['true_counts'])
    log.info("    Pred dist:       %s", metrics['pred_counts'])
    log.info("    Confusion: TP_safe=%d FP_safe=%d FN_safe=%d",
             metrics['tp_safe'], metrics['fp_safe'], metrics['fn_safe'])
    log.info("               TP_unsafe=%d FP_unsafe=%d FN_unsafe=%d",
             metrics['tp_unsafe'], metrics['fp_unsafe'], metrics['fn_unsafe'])

    # Show some misclassifications
    log.info("\n  Sample false negatives (true=unsafe, pred=safe):")
    fn_count = 0
    for rec, true_l, pred_l in zip(val_records, val_labels_true, val_preds):
        if true_l == "unsafe" and pred_l == "safe" and fn_count < 5:
            trace_preview = rec["reasoning_trace"][:150].replace("\n", " ")
            log.info('    [%s] len=%d q="%s" t="%s"',
                     rec['id'], len(rec['reasoning_trace']), rec['query'][:60], trace_preview)
            fn_count += 1

    log.info("\n  Sample false positives (true=safe, pred=unsafe):")
    fp_count = 0
    for rec, true_l, pred_l in zip(val_records, val_labels_true, val_preds):
        if true_l == "safe" and pred_l == "unsafe" and fp_count < 5:
            trace_preview = rec["reasoning_trace"][:150].replace("\n", " ")
            jacc = jaccard_similarity(content_words(rec['query']), content_words(rec['reasoning_trace']))
            log.info('    [%s] len=%d jacc=%.4f q="%s" t="%s"',
                     rec['id'], len(rec['reasoning_trace']), jacc, rec['query'][:60], trace_preview)
            fp_count += 1

    # --- Test predictions ---
    log.info("\n\n--- Test Predictions ---")
    test_records = load_jsonl(TEST_FILE)
    log.info("  Loaded %d test records", len(test_records))

    # Generate predictions
    test_results = []
    for rec in test_records:
        label, detailed = classify_trace(rec["query"], rec["reasoning_trace"])
        test_results.append({
            "id": rec["id"],
            "label": label,
            "detailed_label": detailed,
        })

    # Count distribution
    from collections import Counter
    label_counts = Counter(r["label"] for r in test_results)
    log.info("  Prediction distribution: %s", dict(label_counts))
    for lbl in ["safe", "unsafe", "potentially_unsafe"]:
        if lbl in label_counts:
            log.info("    %s: %d (%.1f%%)", lbl, label_counts[lbl],
                     label_counts[lbl] / len(test_results) * 100)

    # Write submission CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "submission_heuristic.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "label", "detailed_label"])
        for r in test_results:
            writer.writerow([r["id"], r["label"], r["detailed_label"]])

    log.info("\n  Submission written to: %s", output_path)
    log.info("  Total rows: %d", len(test_results))

    # Sanity check: show a few predictions
    log.info("\n  Sample predictions:")
    for r in test_results[:10]:
        rec = next(t for t in test_records if t["id"] == r["id"])
        trace_preview = rec["reasoning_trace"][:80].replace("\n", " ")
        log.info("    %s: %-20s detailed=%-40s t=\"%s\"",
                 r['id'], r['label'], r['detailed_label'][:40], trace_preview)

    log.info("\n" + "=" * 70)
    log.info("  DONE. Validation macro F1 = %.4f", metrics['macro_f1'])
    log.info("=" * 70)


if __name__ == "__main__":
    main()
