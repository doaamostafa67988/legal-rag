"""strip_citations on the answer formats Groq actually produced (taken from the faithfulness cache).

The judge must not see citation boilerplate as a claim, and must still see real claims.
"""

import pytest

from legal_rag.faithfulness import strip_citations

LABEL_LINES = [
    ("نعم، يجب تنفيذ العقد بطريق تتفق مع ما يوجبه حسن النية.\n\n**المادة المعتمدة:** المادة 148 (الفقرة 1).", "148"),
    ("مدة تقادم دعوى التعويض هي **ثلاث سنوات**.\n\n**الاعتماد:** المادة 172 (الفقرة 1).", "172"),
    ("سعر الفائدة **4%** في المسائل المدنية.\n\n**المادة المعتمدة:** المادة 226.", "226"),
]  # fmt: skip


@pytest.mark.parametrize(("answer", "number"), LABEL_LINES)
def test_label_lines_are_removed(answer, number):
    out = strip_citations(answer)
    assert number not in out
    assert "المعتمدة" not in out and "الاعتماد" not in out


def test_markdown_bold_does_not_block_the_lead_pattern():
    out = strip_citations(
        "بناءً على **المادة 1** (الفقرة الثانية)، إذا لم يوجد نص تشريعي يحكم القاضي بالعرف."
    )
    assert "المادة 1" not in out
    assert out.startswith("إذا لم يوجد نص")


def test_lead_with_a_paragraph_marker_is_removed():
    out = strip_citations(
        "بناءً على المادة 44، فقرة (2)، فإن سن الرشد هي **إحدى وعشرون سنة ميلادية كاملة**."
    )
    assert "المادة 44" not in out
    assert "إحدى وعشرون سنة ميلادية كاملة" in out


def test_parenthetical_with_a_paragraph_is_removed():
    out = strip_citations("يجوز له أن يطالب بالفسخ (المادة 157، الفقرة 1).")
    assert "157" not in out and "يجوز له أن يطالب بالفسخ" in out


def test_self_reference_is_removed_but_the_quoted_text_stays():
    answer = 'اعتمدت في إجابتي على **المادة 163** التي تنص صراحةً على أن: "كل خطأ سبب ضررا للغير يلزم من ارتكبه بالتعويض".'
    out = strip_citations(answer)
    assert "اعتمدت" not in out and "المادة 163" not in out
    assert "كل خطأ سبب ضررا للغير يلزم من ارتكبه بالتعويض" in out


@pytest.mark.parametrize(
    ("answer", "kept"),
    [
        (
            'فقد نصت **المادة 141** (الفقرة الأولى) صراحةً على أن البطلان "لا يزول بالإجازة".',
            "المادة 141",
        ),
        (
            "كما تنص **المادة 200** على أن القاضي يقدر عند عدم النص ما إذا كان هناك التزام طبيعي.",
            "المادة 200",
        ),
    ],
)
def test_claims_that_name_an_article_are_kept(answer, kept):
    assert kept in strip_citations(answer)
