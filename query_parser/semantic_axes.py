"""
Semantic Axes
-------------
Defines abstract intent dimensions using anchor phrases.
These are NOT keywords — they are semantic centroids.
"""

SEMANTIC_AXES = {
    "emotion": [
        # Dramatic / sad
        "emotionally intense",
        "sad and touching",
        "heartfelt drama",
        "deep emotional journey",
        "moving story",
        # Happy / joyful
        "funny and entertaining",
        "joyful and lighthearted",
        "feel-good and uplifting",
        # Romantic
        "romantic and tender",
        # Nostalgic
        "nostalgic and sentimental",
        # Tense / dark
        "thrilling and suspenseful",
        "dark and disturbing",
    ],
    "genre": [
        "action packed movie",
        "romantic film",
        "psychological thriller",
        "science fiction movie",
        "light hearted comedy",
        "animated family movie",
        "documentary film",
        "biographical drama",
        "mystery and detective story",
    ],
    "tone": [
        "dark and gritty",
        "uplifting and inspiring",
        "slow paced and artistic",
        "fast paced blockbuster",
        "quirky and offbeat",
        "dry and witty",
        "feel-good and wholesome",
    ]
}
