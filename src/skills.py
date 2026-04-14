import re


def extract_skills_from_text(text, skill_vocab):
    text_lower = text.lower()
    found = set()

    for skill in skill_vocab:
        skill_lower = skill.lower()
        pattern = r"\b" + re.escape(skill_lower) + r"\b"

        if re.search(pattern, text_lower):
            found.add(skill_lower)

    return list(found)