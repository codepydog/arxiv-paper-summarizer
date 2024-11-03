"""Translation utilities for the summarizer."""

from arxiv_paper_summarizer.prompt_function import openai_prompt


@openai_prompt(
    ("system", "You are an AI research assistant."),
    (
        "user",
        "##Task\n"
        "Translate the provided note, which is enclosed in triple backticks, into the specified language, {language}. "
        "Follow these rules for translation:\n"
        "- Preserve the original meaning as closely as possible.\n"
        "- Use terminology commonly used by data scientists and AI researchers.\n"
        "- Avoid over-translation; keep terms intact where applicable.\n"
        "## Note Content\n```{text}```\n\n",
    ),
    model_name="gpt-4o",
)
def translate_text(text: str, language: str): ...  # type: ignore[empty-body]
