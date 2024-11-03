"""Translation utilities for the summarizer."""

from arxiv_paper_summarizer.prompt_function import openai_prompt


@openai_prompt(
    ("system", "You are an AI research assistant."),
    (
        "user",
        "##Task\n"
        "Translate the provided note into the specified language, {language}. "
        "Follow these rules for translation:\n"
        "- Preserve the original meaning as closely as possible.\n"
        "- Use terminology commonly used by data scientists and AI researchers.\n"
        "- Avoid over-translation; keep terms intact where applicable.\n\n"
        "Here is the note:\n"
        "{text}",
    ),
    model_name="gpt-4o",
)
def translate_text(text: str, language: str): ...  # type: ignore[empty-body]


@openai_prompt(
    ("system", "You are an AI research assistant."),
    (
        "user",
        "##Task\n"
        "Translate the provided quote into the specified language, {language}. "
        "Follow these rules for translation:\n"
        "- Do NOT translate the quote itself.\n"
        "- Add a translation of the quote below the original quote.\n"
        "- Preserve the original meaning as closely as possible.\n"
        "- Use terminology commonly used by data scientists and AI researchers.\n"
        "- Avoid over-translation; keep terms intact where applicable.\n\n"
        "## Response Format\n"
        "> 'Original quote text here'\n\n"
        "Translated quote text here. (Do not include the quote marks.)\n\n"
        "Here is the quote:\n"
        "{text}",
    ),
    model_name="gpt-4o",
)
def translate_quote(text: str, language: str): ...  # type: ignore[empty-body]
