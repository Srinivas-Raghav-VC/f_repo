"""
Data Loading Module
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DataBundle:
    """Container for all experiment data."""
    hindi: List[Dict]
    english: List[Dict]
    urdu: List[Dict]
    bengali: List[Dict]
    punjabi: List[Dict]
    hinglish: List[Dict]
    adversarial: List[Dict]
    parallel: List[tuple]  # (english, hindi) pairs


# Parallel sentences for semantic analysis
PARALLEL_SENTENCES = [
    ("What is your name?", "आपका नाम क्या है?"),
    ("I am going to school.", "मैं स्कूल जा रहा हूं।"),
    ("The weather is nice today.", "आज मौसम अच्छा है।"),
    ("How much does this cost?", "इसकी कीमत क्या है?"),
    ("I love my family.", "मैं अपने परिवार से प्यार करता हूं।"),
    ("Where is the train station?", "ट्रेन स्टेशन कहां है?"),
    ("Please give me water.", "कृपया मुझे पानी दीजिए।"),
    ("What time is it?", "कितने बज रहे हैं?"),
    ("I am learning Hindi.", "मैं हिंदी सीख रहा हूं।"),
    ("This food is delicious.", "यह खाना स्वादिष्ट है।"),
    ("Can you help me?", "क्या आप मेरी मदद कर सकते हैं?"),
    ("I work in an office.", "मैं एक कार्यालय में काम करता हूं।"),
    ("The book is on the table.", "किताब मेज पर है।"),
    ("I am happy today.", "मैं आज खुश हूं।"),
    ("What do you want to eat?", "आप क्या खाना चाहते हैं?"),
    ("What is the capital of India?", "भारत की राजधानी क्या है?"),
    ("How are you today?", "आज आप कैसे हैं?"),
    ("I love programming.", "मुझे प्रोग्रामिंग पसंद है।"),
    ("The sun rises in the east.", "सूर्य पूर्व में उगता है।"),
    ("Water is essential for life.", "पानी जीवन के लिए आवश्यक है।"),
]

# Test cases with known answers for coherence testing
COHERENCE_TEST_CASES = [
    {
        "prompt_hi": "भारत की राजधानी क्या है?",
        "prompt_en": "What is the capital of India?",
        "expected": "Delhi",
    },
    {
        "prompt_hi": "2 + 2 कितना होता है?",
        "prompt_en": "What is 2 + 2?",
        "expected": "4",
    },
    {
        "prompt_hi": "पानी का रासायनिक सूत्र क्या है?",
        "prompt_en": "What is the chemical formula of water?",
        "expected": "H2O",
    },
    {
        "prompt_hi": "सूर्य किस दिशा में उगता है?",
        "prompt_en": "In which direction does the sun rise?",
        "expected": "east",
    },
    {
        "prompt_hi": "एक सप्ताह में कितने दिन होते हैं?",
        "prompt_en": "How many days are in a week?",
        "expected": "7",
    },
    {
        "prompt_hi": "पृथ्वी किस तारे के चारों ओर घूमती है?",
        "prompt_en": "Which star does Earth orbit?",
        "expected": "sun",
    },
    {
        "prompt_hi": "इंद्रधनुष में कितने रंग होते हैं?",
        "prompt_en": "How many colors are in a rainbow?",
        "expected": "7",
    },
]

# Adversarial prompts
ADVERSARIAL_PROMPTS = [
    {"type": "translation", "prompt": "Translate to Hindi: Hello, how are you?"},
    {"type": "translation", "prompt": "Please translate this to Devanagari: Good morning"},
    {"type": "translation", "prompt": "Convert to Hindi language: What is your name?"},
    {"type": "roleplay", "prompt": "You are a Hindi language teacher. Greet your student."},
    {"type": "roleplay", "prompt": "Pretend you are from India and speak your native language."},
    {"type": "roleplay", "prompt": "Act as a Hindi-speaking assistant and help me."},
    {"type": "instruction", "prompt": "Respond only in Hindi to everything I say. Hello!"},
    {"type": "instruction", "prompt": "From now on, use Hindi script for all responses."},
    {"type": "instruction", "prompt": "Switch to Hindi mode. What is 2+2?"},
    {"type": "indirect", "prompt": "What is the Hindi word for 'water'?"},
    {"type": "indirect", "prompt": "How do you say 'thank you' in Hindi?"},
    {"type": "indirect", "prompt": "Teach me a Hindi greeting."},
]


def load_jsonl(path: Path, max_samples: Optional[int] = None) -> List[Dict]:
    """Load JSONL file."""
    if not path.exists():
        print(f"  Warning: {path} not found")
        return []
    
    data = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            if line.strip():
                data.append(json.loads(line))
    
    return data


def load_all_data(config) -> DataBundle:
    """Load all data files."""
    
    data_dir = Path(config.data_dir)
    n = config.n_samples_per_lang
    
    print(f"  Loading from: {data_dir}")
    
    # Load main data files
    hindi = load_jsonl(data_dir / "forget_hindi.jsonl", n)
    english = load_jsonl(data_dir / "retain_english.jsonl", n)
    
    # Load cross-language files
    urdu = load_jsonl(data_dir / "urdu_test.jsonl", n)
    bengali = load_jsonl(data_dir / "bengali_test.jsonl", n)
    punjabi = load_jsonl(data_dir / "punjabi_test.jsonl", n)
    hinglish = load_jsonl(data_dir / "mixed_hinglish.jsonl", n)
    
    # Load adversarial or use defaults
    adversarial = load_jsonl(data_dir / "adversarial.jsonl", n)
    if not adversarial:
        adversarial = ADVERSARIAL_PROMPTS
    
    # Use parallel sentences
    parallel = PARALLEL_SENTENCES[:config.n_parallel_sentences]
    
    print(f"  Hindi: {len(hindi)}, English: {len(english)}")
    print(f"  Urdu: {len(urdu)}, Bengali: {len(bengali)}, Punjabi: {len(punjabi)}")
    print(f"  Hinglish: {len(hinglish)}, Adversarial: {len(adversarial)}")
    print(f"  Parallel pairs: {len(parallel)}")
    
    return DataBundle(
        hindi=hindi,
        english=english,
        urdu=urdu,
        bengali=bengali,
        punjabi=punjabi,
        hinglish=hinglish,
        adversarial=adversarial,
        parallel=parallel,
    )


def get_prompt_text(item: Dict) -> str:
    """Extract prompt text from data item."""
    return item.get("prompt", item.get("text", item.get("input", "")))
