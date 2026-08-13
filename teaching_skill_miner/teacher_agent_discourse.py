"""Shared deterministic learner-discourse safety classification.

The classifier is a conservative safety floor used before model planning.  It
does not decide whether an academic answer is correct.  Its only job is to
preserve learner control signals which must survive provider failure or an
incorrect model label: questions, requests for explanation, explicit reports
of confusion, and requests that risk disclosing the active exercise solution.

Keeping this logic in one dependency-free module prevents the live planner,
memory replay, and context compactor from silently disagreeing about the same
utterance.  The patterns deliberately cover common Chinese and English chat
forms, including sentences without a final question mark.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


_MAX_DISCOURSE_CHARS = 600


@dataclass(frozen=True, slots=True)
class LearnerDiscourse:
    """Content-free routing facts derived from one bounded learner message."""

    is_question: bool
    clarification_kind: str | None
    explicit_confusion: bool
    solution_risk: str


_QUESTION_PUNCTUATION_RE = re.compile(r"[?？]")
_QUESTION_REQUEST_RE = re.compile(
    r"(?:^|[，。！？；,.!?;\s])(?:请问|能不能|能否|可不可以|可以(?:再)?|"
    r"你能|could\s+you|can\s+you|would\s+you)",
    re.IGNORECASE,
)
_INTERROGATIVE_RE = re.compile(
    r"(?:什么|什么意思|为何|为什么|怎么|怎样|如何|哪(?:个|些|几|一步|里)|"
    r"几(?:个|种|步|部分)|多少|是否|是不是|能否|会不会|"
    r"\bwhat\b|\bwhy\b|\bhow\b|\bwhich\b|\bwhere\b|\bwhen\b|"
    r"\bcan\s+you\b|\bcould\s+you\b)",
    re.IGNORECASE,
)

_CONFUSION_RE = re.compile(
    r"(?:^|[，。！？；,.!?;\s])(?:我)?(?:还是|真的|完全|有点|有些|暂时|一直)?\s*"
    r"(?:不会|不知道|不清楚|没懂|没明白|没搞懂|不懂|不理解|不太理解|"
    r"跟不上|没跟上|有点懵|很懵|糊涂|卡住(?:了)?|想不起来|分不清|"
    r"看不懂|听不懂|答不出|说不出)"
    r"|\bi\s*(?:am|'m)\s+(?:lost|confused|stuck)\b"
    r"|\bi\s+(?:do\s+not|don't|cannot|can't)\s+(?:understand|follow|remember|tell)\b"
    r"|\bi\s+(?:have\s+)?no\s+idea\b"
    r"|\b(?:this|that|it)\s+(?:went|goes)\s+over\s+my\s+head\b"
    r"|(?<!不)(?:有点|有些|完全|一直)?\s*(?:跟不上|没跟上|很懵|有点懵)",
    re.IGNORECASE,
)
_NEGATED_CONFUSION_RE = re.compile(
    r"(?:不是|并非|不算)(?:跟不上|没跟上|不懂|不理解|不会|不知道|不清楚|迷糊|懵)"
    r"|\b(?:not|never)\s+(?:lost|confused|stuck)\b",
    re.IGNORECASE,
)
_CONTRASTED_CONFUSION_RE = re.compile(
    r"(?:而是|但|但是|不过|可是|but|however).{0,20}"
    r"(?:跟不上|没跟上|不懂|不理解|不会|不知道|不清楚|迷糊|懵|lost|confused|stuck)",
    re.IGNORECASE,
)

_FINAL_SOLUTION_RE = re.compile(
    r"(?:直接)?(?:告诉|给|说出|公布).{0,10}(?:答案|解法|结果)"
    r"|(?:最终|完整|标准|正确)(?:答案|解法|结果)"
    r"|(?:这题|这道题|当前题|这个练习|当前任务).{0,12}"
    r"(?:怎么做|如何做|解一下|做完|写完)"
    r"|(?:把|帮我).{0,14}(?:答案|解法|代码|题).{0,10}(?:写出|写完|做完)"
    r"|\b(?:give|tell|show)\s+me\s+(?:the\s+)?(?:final|complete|correct)?\s*"
    r"(?:answer|solution|code)\b"
    r"|\b(?:solve|finish|do)\s+(?:this|the)\s+(?:problem|exercise|task)\s+for\s+me\b",
    re.IGNORECASE,
)
_CURRENT_TASK_RE = re.compile(
    r"(?:这题|这道题|当前题|这个练习|当前任务|这一步|下一步|这里|此处|"
    r"这个式子|这段推导|这段证明|这段代码|我的答案)"
    r"|\b(?:this|current|next)\s+(?:problem|exercise|task|step|equation|proof|code)\b",
    re.IGNORECASE,
)

_SYMBOL_RE = re.compile(
    r"(?:表示|代表)(?:的)?(?:是)?什么|"
    r"(?:符号|字母|变量|参数|下标).{0,16}(?:含义|作用)(?:是)?什么|"
    r"(?:符号|字母|变量|参数|下标).{0,16}(?:怎么|如何)读|"
    r"\bwhat\s+does\s+.{1,40}\s+mean\b|"
    r"\bhow\s+(?:do|should)\s+(?:i|we|you)\s+(?:read|say|pronounce)\b",
    re.IGNORECASE,
)
_COMPOSITION_RE = re.compile(
    r"(?:分|有|包括|包含|由|拆成|拆分成|拆开).{0,10}(?:哪几|哪些|几|多少)(?:个)?"
    r"(?:种|部分|方面|类|要素|成分|内容|东西)|"
    r"(?:哪几|哪些|几|多少)(?:个)?(?:种|部分|方面|类|要素|成分)|"
    r"由(?:哪些|什么).{0,14}(?:组成|构成)|(?:包括|包含)(?:哪些|什么)|"
    r"\bwhat\s+(?:are|is)\s+(?:the\s+)?(?:parts?|components?|elements?)\s+of\b|"
    r"\bhow\s+many\s+(?:parts?|components?|steps?)\b|"
    r"\bbreak\s+.{0,50}\s+(?:down|into)\b",
    re.IGNORECASE,
)
_RATIONALE_RE = re.compile(
    r"(?:为什么|为何|原因(?:是)?什么)|\bwhy\b|\bwhat\s+is\s+the\s+reason\b",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"(?:有什么|有何)(?:区别|差别|不同)|"
    r"(?:区别|差别|不同)(?:是什么|在哪|有哪些)|(?:怎么|如何)区分|"
    r"\b(?:difference|distinction)\s+between\b|\bhow\s+(?:are|is).{1,60}\bdifferent\b",
    re.IGNORECASE,
)
_EXAMPLE_RE = re.compile(
    r"(?:举|给).{0,8}(?:个|一个)?(?:例子|示例)|(?:例子|示例)(?:是什么|有哪些)|"
    r"\b(?:give|show|provide)\s+(?:me\s+)?(?:an?\s+)?examples?\b",
    re.IGNORECASE,
)
_PROCEDURE_RE = re.compile(
    r"(?:步骤|流程)(?:是什么|有哪些|有哪几|有几)|"
    r"(?:怎么|如何)(?:做|操作|计算|推导|判断|选择|设置|使用|开始|分解|实现|运行|工作)|"
    r"\bhow\s+(?:does|do|is|are|can|should)\b.{0,80}\b(?:work|derive|calculate|use|start)\b|"
    r"\bwhat\s+(?:are|is)\s+(?:the\s+)?steps?\b|\bhow\s+to\b",
    re.IGNORECASE,
)
_DEFINITION_RE = re.compile(
    r"(?:什么是|是什么意思|是什么概念|什么意思|定义(?:是)?什么|含义是什么|"
    r"指(?:的)?是什么|(?:怎么|如何)理解|(?:原理|机制|作用)(?:是)?什么)|"
    r"\bwhat\s+is\b|\bwhat\s+does\b.{1,60}\bmean\b|\bdefine\b",
    re.IGNORECASE,
)
_ELABORATION_RE = re.compile(
    r"(?:展开|详细|具体).{0,12}(?:讲|说|解释|说明)(?:一下|一点)?|"
    r"(?:展开|详细说明|具体说明)(?:一下|一点)(?:这里|这个|这一步|这部分)?|"
    r"(?:再|重新).{0,8}(?:解释|讲讲|说明)|(?:讲清楚|说清楚)|"
    r"\b(?:unpack|elaborate|explain|go\s+over)\b",
    re.IGNORECASE,
)

_DECLARATIVE_KNOWLEDGE_RE = re.compile(
    r"^(?:我|我们)(?:已经|现在|基本)?(?:知道|明白|理解|能(?:够)?解释|可以解释).{0,180}$|"
    r"^(?:我|我们).{0,50}(?:把|将).{0,50}(?:分成|分为).{0,30}(?:个|部分)(?:了)?[！!。.]*$|"
    r"^.{1,120}(?:只表示|只用于|不需要|无需|一定|必须|会导致).{0,180}$|"
    r"^.{1,100}(?:就是|等于|意味着|因为|所以|一定|必须|会导致).{1,180}$|"
    r"^(?:i|we)\s+(?:already\s+)?(?:know|understand|can\s+explain)\b.{0,180}$|"
    r"^(?!\s*(?:what|why|how|which|where|when)\b).{1,100}\b"
    r"(?:is|are|means?|equals?|because|therefore|must|causes?)\b.{0,180}$",
    re.IGNORECASE,
)
_CONTRAST_RE = re.compile(
    r"(?:但|但是|不过|可是|although|but|however).{0,140}"
    r"(?:为什么|为何|什么意思|表示什么|怎么|如何|\bwhy\b|\bwhat\b|\bhow\b)",
    re.IGNORECASE,
)


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).strip().split())


def classify_learner_discourse_text(value: str) -> LearnerDiscourse:
    """Return shared, conservative routing facts for one learner utterance."""

    text = _normalized_text(value)
    if not text or len(text) > _MAX_DISCOURSE_CHARS:
        return LearnerDiscourse(False, None, False, "none")

    final_solution = bool(_FINAL_SOLUTION_RE.search(text))
    current_task = bool(_CURRENT_TASK_RE.search(text))
    explicit_confusion = bool(_CONFUSION_RE.search(text)) and (
        not _NEGATED_CONFUSION_RE.search(text)
        or bool(_CONTRASTED_CONFUSION_RE.search(text))
    )

    clarification_kind: str | None = None
    for kind, pattern in (
        ("symbol_meaning", _SYMBOL_RE),
        ("composition", _COMPOSITION_RE),
        ("rationale", _RATIONALE_RE),
        ("comparison", _COMPARISON_RE),
        ("example_request", _EXAMPLE_RE),
        ("procedure", _PROCEDURE_RE),
        ("definition", _DEFINITION_RE),
        # An underspecified request to unpack the current explanation is still
        # explanation-seeking.  Mapping it to definition lets the existing
        # answer-first grounding contract resolve the visible subject.
        ("definition", _ELABORATION_RE),
    ):
        if pattern.search(text):
            clarification_kind = kind
            break

    if final_solution:
        solution_risk = "final_solution"
    elif current_task and clarification_kind in {"procedure", "rationale"}:
        solution_risk = "current_step"
    else:
        solution_risk = "none"

    declaration_without_question = bool(
        _DECLARATIVE_KNOWLEDGE_RE.search(text)
        and not _CONTRAST_RE.search(text)
        and not _QUESTION_PUNCTUATION_RE.search(text)
        and not _QUESTION_REQUEST_RE.search(text)
    )
    is_question = bool(
        final_solution
        or _QUESTION_PUNCTUATION_RE.search(text)
        or _QUESTION_REQUEST_RE.search(text)
        or (clarification_kind is not None and not declaration_without_question)
        or (_INTERROGATIVE_RE.search(text) and not declaration_without_question)
    )
    if declaration_without_question:
        clarification_kind = None
    if final_solution:
        # A request for the active solution remains a learner question, but it
        # is not an answer-first conceptual clarification.
        clarification_kind = None

    return LearnerDiscourse(
        is_question=is_question,
        clarification_kind=clarification_kind,
        explicit_confusion=explicit_confusion,
        solution_risk=solution_risk,
    )
