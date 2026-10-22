from legal_rag.data import Article
from legal_rag.validate_data import validate


def art(n, ar="نص", en="text", repealed=False, en_missing=False, citation=None):
    return Article(
        article_number=n, text_ar=ar, text_en=en, is_repealed=repealed,
        citation=f"Art. {n}" if citation is None else citation,
        book="", chapter="", section="", topic="", source_page=1, en_missing=en_missing,
    )  # fmt: skip


def corpus(count=5):
    return [art(n) for n in range(1, count + 1)]


def test_a_good_corpus_has_no_problems():
    assert validate(corpus(), min_articles=5) == []


def test_too_few_articles():
    assert "only 5 articles" in validate(corpus(), min_articles=10)[0]


def test_repeated_article_numbers():
    articles = corpus() + [art(3)]
    assert any("more than once: [3]" in p for p in validate(articles, min_articles=5))


def test_an_active_article_needs_arabic_text():
    articles = corpus() + [art(9, ar="  ")]
    assert any("no Arabic text: [9]" in p for p in validate(articles, min_articles=5))


def test_a_repealed_article_may_be_empty():
    articles = corpus() + [art(9, ar="", en="", repealed=True)]
    assert validate(articles, min_articles=5) == []


def test_missing_english_needs_the_parser_flag():
    assert any("no English text: [9]" in p for p in validate(corpus() + [art(9, en="")], 5))
    assert validate(corpus() + [art(9, en="", en_missing=True)], 5) == []


def test_every_article_needs_a_citation():
    articles = corpus() + [art(9, citation=" ")]
    assert any("no citation: [9]" in p for p in validate(articles, min_articles=5))
