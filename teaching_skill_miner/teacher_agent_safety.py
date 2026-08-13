"""Deterministic student-safety and academic-integrity boundary.

This module is intentionally small and conservative.  It does not diagnose a
learner or replace emergency, medical, legal, safeguarding, or school policy
professionals.  It identifies a bounded set of high-risk disclosures and
requests before they reach the teaching planner, then returns a content-free
audit contract and a fixed response policy.  Matches are never mastery
evidence.
"""

from __future__ import annotations

from hashlib import sha256
import re
from typing import Any, Mapping
import unicodedata


LEARNER_SAFETY_CONTRACT_SCHEMA = "teaching_skill_miner.learner_safety_contract.v1"
ASSISTANT_OUTPUT_SAFETY_SCHEMA = (
    "teaching_skill_miner.assistant_output_safety_contract.v1"
)
SAFETY_FOLLOW_UP_STATUSES = frozenset({"unknown", "explicitly_safe", "still_unsafe"})

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_FIRST_PERSON_SELF_HARM = re.compile(
    r"(?:我|本人).{0,24}(?:想死|不想活|活不下去|撑不下去|结束生命|结束这一切|"
    r"自杀|自残|割腕|吞药|跳下去|跳楼|上吊|伤害自己|杀了自己|永远消失)"
    r"|(?:活着|继续活).{0,8}(?:没意思|没有意义)"
    r"|\bI(?:(?:'m| am)\s+(?:going|planning|intending|about)\s+to|"
    r"(?:'ll| will)|"
    r"\s+(?:want|plan|intend)\s+to)\s+"
    r"(?:commit suicide|die|kill myself|hurt myself|end my life|"
    r"take my own life|end it all|cut myself|(?:cut|slit) my wrists?|"
    r"hang myself|(?:take|swallow)\s+(?:all\s+(?:of\s+)?my\s+pills?|"
    r"the\s+whole\s+bottle)|(?:step|jump)\s+off\s+(?:the\s+)?"
    r"(?:roof|bridge|building))\b"
    r"|\bI\s+have\s+(?:a\s+)?plan\s+to\s+(?:commit suicide|die|"
    r"kill myself|hurt myself|end my life|take my own life|hang myself)\b"
    r"|\bI(?:'m| am)\s+(?:suicidal|hurting myself|better off dead)\b"
    r"|\bI\s+(?:do not|don't)\s+want\s+to\s+(?:live|be alive)\b"
    r"|\bI\s+(?:do not|don't)\s+want\s+to\s+(?:wake\s+up|see\s+tomorrow)\b"
    r"|\bI\s+(?:cannot|can't)\s+(?:go on|keep living)\b"
    r"|\bI\s+(?:saved|have\s+saved|collected|have\s+collected)\s+"
    r"(?:all\s+)?(?:of\s+)?(?:my\s+)?pills?.{0,24}\b(?:tonight|tomorrow|now)\b"
    r"|\bI\s+(?:(?:am\s+going|plan|intend)\s+to|will)\s+"
    r"(?:step|jump)\s+off\s+(?:the\s+)?(?:roof|bridge|building)\b"
    r"|(?:我|本人).{0,24}(?:希望|想要|最好).{0,12}(?:别再醒来|不再醒来|消失|从楼上跳下去)"
    r"|(?:我|本人).{0,16}不想.{0,10}(?:明天|以后)?(?:再)?醒来"
    r"|(?:我|本人).{0,20}(?:把)?(?:药|药片).{0,10}(?:攒|存|准备)(?:好|齐|下)?了?"
    r".{0,18}(?:今晚|明天|现在).{0,12}(?:结束|不再醒来|不醒来|永远消失)"
    r"|(?:没有我|我不在了).{0,12}(?:大家|他们|家人).{0,8}(?:会更好|更轻松)"
    r"|\bI(?:'m| am)?\s*(?:thinking|thought)\s+(?:about|of)\s+(?:ending my life|suicide|hurting myself)\b"
    r"|\bI\s+wish\s+I\s+(?:would not|wouldn't|did not|didn't)\s+wake\s+up\b"
    r"|\b(?:everyone|my family)\s+would\s+be\s+better\s+(?:off\s+)?without\s+me\b"
    r"|(?:我|本人).{0,16}(?:从|去).{0,8}(?:天台|楼顶|桥上).{0,8}"
    r"(?:跳下(?:去|来)?|一跃而下)",
    re.IGNORECASE,
)
_IMMINENCE = re.compile(
    r"现在|马上|今晚|明天|很快|已经|计划|遗书|刀|药|跳楼|上吊|枪"
    r"|\b(?:now|tonight|tomorrow|soon|after class|already|plan|pills?|knife|gun|bridge|rope)\b"
    r"|放学后",
    re.IGNORECASE,
)
_ABUSE_DISCLOSURE = re.compile(
    r"(?:老师|教练|家长|家里人|爸爸|妈妈|继父|继母|叔叔|阿姨|亲戚|成年人|他|她)"
    r".{0,20}(?:打我|虐待我|摸我|猥亵我|强迫我|威胁我|不让我说|让我保密|拍我裸照)"
    r"|\b(?:teacher|coach|parent|adult|uncle|aunt|stepfather|stepmother|he|she|they)"
    r".{0,50}(?:hits?|abuses?|touches?|forces?|threatens?|makes? me keep it secret|"
    r"takes? nude photos? of)\s+me\b"
    r"|(?:家里|学校|训练时).{0,12}(?:有人|大人|老师|教练).{0,20}"
    r"(?:摸我的私处|逼我脱衣服|逼我发裸照|让我保密|不许我告诉别人)"
    r"|\b(?:someone|an adult).{0,24}(?:at home|at school|during practice).{0,40}"
    r"(?:touches my private parts|makes me undress|asks me for nude photos|tells me to keep it secret)\b",
    re.IGNORECASE,
)
_BULLYING_DISCLOSURE = re.compile(
    r"(?:同学|同桌|他们|她们|群里的人).{0,20}"
    r"(?:霸凌我|欺负我|威胁我|勒索我|围堵我|造谣我|发我的照片|网暴我)"
    r"|\b(?:classmates?|students?|they|people online).{0,50}"
    r"(?:bully|threaten|blackmail|harass|spread rumors about|post photos of)\s+me\b"
    r"|(?:我|本人).{0,12}(?:每天|一直|经常)?被(?:同学|他们|她们|群里的人).{0,18}"
    r"(?:锁在厕所|推搡|殴打|辱骂|排挤|勒索|围堵|威胁|偷拍视频|网暴)"
    r"|\bI(?:'m| am)?\s*(?:being|keep getting)\s+(?:locked in|pushed|hit|excluded|blackmailed|harassed)"
    r".{0,30}\b(?:classmates?|students?|people online|them)\b",
    re.IGNORECASE,
)
_MINOR_SEXUAL_REQUEST = re.compile(
    r"(?:未成年|儿童|小学生|初中生).{0,20}(?:色情|裸照|性行为|成人视频)"
    r"|(?:色情|裸照|性行为).{0,20}(?:未成年|儿童|小学生|初中生)"
    r"|\b(?:child|minor|underage).{0,30}(?:porn|nude|sexual|sex|explicit)\b"
    r"|\b(?:groom|sextort).{0,30}(?:child|minor|student)\b"
    r"|(?:生成|画|制作|寻找|发我).{0,20}(?:[1-9]|1[0-7])岁.{0,12}(?:裸体|裸照|色情|性行为)"
    r"|\b(?:generate|draw|make|find|show).{0,30}(?:[1-9]|1[0-7])[- ]year[- ]old.{0,30}"
    r"(?:nude|naked|sexual|explicit|porn)\b",
    re.IGNORECASE,
)
_URGENT_MEDICAL = re.compile(
    r"(?:我|我的?孩子|我的学生).{0,12}(?:胸痛|呼吸困难|昏迷|抽搐|大量出血|吃了很多药|药物过量)"
    r"|\b(?:I|my child|the student).{0,40}(?:chest pain|cannot breathe|can't breathe|"
    r"unconscious|seizure|overdose|heavy bleeding|swallowed (?:pills|medicine))\b"
    r"|(?:我|我的?孩子|我的学生).{0,16}(?:刚|已经).{0,8}(?:吞了|吃了).{0,8}"
    r"(?:[一二三四五六七八九十百0-9]+片|整瓶).{0,8}(?:药|药片)"
    r"|\b(?:I|my child|the student).{0,30}(?:just|already).{0,12}"
    r"(?:swallowed|took).{0,12}(?:a whole bottle|[0-9]+ pills?)\b",
    re.IGNORECASE,
)
_HIGH_RISK_ADVICE = re.compile(
    r"(?:该不该|能不能|怎么).{0,8}(?:停药|加药|换药|签认罪书|销毁证据|逃避警方)"
    r"|\b(?:should I|how do I).{0,30}(?:stop medication|change dosage|destroy evidence|evade police|sign a confession)\b",
    re.IGNORECASE,
)
_CHEATING_REQUEST = re.compile(
    r"(?:替我|帮我).{0,10}(?:考试|代考|作弊|抄袭|直接写完|绕过监考|伪造实验数据)"
    r"|(?:直接给我|只要)(?:作业|考试|测验|论文).{0,8}(?:答案|成品)"
    r"|\b(?:take my exam|cheat for me|bypass proctor|plagiarize|fabricate data|write my assignment for me)\b",
    re.IGNORECASE,
)
_TARGETED_HATE = re.compile(
    r"(?:所有|这些).{0,12}(?:人|学生|民族|种族|宗教|残疾人|同性恋).{0,8}"
    r"(?:都该死|都很蠢|不配活|应该消失)"
    r"|\b(?:all|those).{0,20}(?:people|students|race|religion|disabled|gay).{0,20}"
    r"(?:should die|are stupid|should disappear|should be hurt|do not deserve to live)\b"
    r"|(?:某个|这个|那个)?(?:民族|种族|宗教|残疾人|同性恋|女性|男性).{0,10}"
    r"(?:天生低等|天生愚蠢|不配受教育|应该被清除)"
    r"|\b(?:that|this)?\s*(?:race|religion|disabled people|gay people|women|men).{0,20}"
    r"(?:(?:is|are) inherently inferior|(?:is|are) born stupid|should be eliminated)\b",
    re.IGNORECASE,
)

_REPORTED_SELF_HARM_EXCLUSION = re.compile(
    r"(?:小说|文学|新闻|案例|课文|材料|历史).{0,24}"
    r"(?:主人公|角色|作者|患者|当事人|有人说|材料中|描述|讨论|研究)"
    r"|\b(?:novel|story|news|case study|lesson|historical source).{0,40}"
    r"(?:character|author|patient|participant|quotation|describes?|discusses?)\b",
    re.IGNORECASE,
)
_PERSONAL_CONTEXT_PIVOT = re.compile(
    r"(?:^|[，。！？；;,.!?])\s*(?:但(?:是)?|可是|然而|不过|其实|而我|至于我|"
    r"我(?:现在|此刻|目前)|but\s+I|however\s*,?\s*I|actually\s*,?\s*I|"
    r"as\s+for\s+me)",
    re.IGNORECASE,
)
_CRITICAL_HATE_EXCLUSION = re.compile(
    r"(?:请|需要|我们来)?(?:批判|反驳|纠正).{0,20}(?:偏见|歧视|说法|论点)"
    r"|(?:这是|属于).{0,8}(?:偏见|歧视|仇恨言论)"
    r"|\b(?:critique|refute|challenge|correct).{0,30}(?:bias|prejudice|stereotype|claim)\b"
    r"|\b(?:this is|it is).{0,12}(?:bias|prejudice|discrimination|hate speech)\b",
    re.IGNORECASE,
)

_OBJECTIVE_SAFETY_EDUCATION = re.compile(
    r"(?:预防|求助|安全教育|风险识别|保护儿童|保护未成年人|举报|报告|法律|伦理|"
    r"案例分析|新闻报道|研究|课程|教材|课堂讨论|历史材料|文学作品|"
    r"prevention|safety education|risk awareness|safeguarding|reporting|"
    r"case study|news report|research|lesson|curriculum|historical source|literature)",
    re.IGNORECASE,
)
_RESOURCE_HARMFUL_SELF_HARM_INSTRUCTION = re.compile(
    r"(?:自杀|自残|割腕|吞药).{0,20}(?:方法|步骤|教程|更有效|最有效|不被发现)"
    r"|(?:方法|步骤|教程).{0,20}(?:自杀|自残|割腕|吞药)"
    r"|\b(?:how to|best way to|instructions? (?:for|to)).{0,30}"
    r"(?:suicide|kill yourself|self[- ]harm|cut yourself)\b",
    re.IGNORECASE,
)
_EXPLICIT_SELF_HARM_DENIAL = re.compile(
    r"(?:我|本人)(?:现在|此刻|目前)?(?:不会|不打算|没有.{0,8}(?:打算|计划|念头))"
    r".{0,8}(?:自杀|自残|伤害自己|结束生命|跳楼|上吊|"
    r"从(?:天台|楼顶|桥上)跳下去)"
    r"|\bI\s+(?:will not|won't|am not going to|have no (?:plan|intention) to)"
    r"\s+(?:die|kill myself|hurt myself|harm myself|end my life|"
    r"take my own life|hang myself)\b",
    re.IGNORECASE,
)

_FOLLOW_UP_EXPLICIT_SAFE = re.compile(
    r"(?:我(?:现在|此刻|目前)?(?:很)?安全|"
    r"我(?:现在|此刻|目前)?不会(?:伤害自己|自杀|自残)|"
    r"我(?:现在|此刻|目前)?没有.{0,10}(?:伤害自己|自杀|自残).{0,8}(?:打算|计划|念头))"
    r"|\bI\s+(?:am|feel)\s+safe\s+(?:now|at the moment)\b"
    r"|\bI\s+(?:will not|won't)\s+(?:hurt|harm|kill)\s+myself\b"
    r"|\bI\s+have\s+no\s+(?:plan|intention)\s+to\s+"
    r"(?:hurt|harm|kill)\s+myself\b",
    re.IGNORECASE,
)
_FOLLOW_UP_TRUSTED_SUPPORT = re.compile(
    r"(?:(?:家长|父母|妈妈|爸爸|老师|辅导员|心理老师|成年人|家人|"
    r"可信任的人|可信任的朋友).{0,12}(?:陪着(?:我)?|在我身边|和我在一起|正在帮助我|"
    r"已经联系|会陪着我)|"
    r"我(?:现在)?和(?:家长|父母|妈妈|爸爸|老师|辅导员|心理老师|家人|"
    r"可信任的人|可信任的朋友)在一起|有人陪着我)"
    r"|\b(?:(?:my|a)\s+)?(?:parent|mother|father|teacher|counselor|trusted adult|"
    r"family member|trusted friend)\s+(?:is|are)\s+(?:here|with me)\b"
    r"|\bI\s+am\s+with\s+(?:my\s+)?(?:parent|mother|father|teacher|counselor|"
    r"a trusted adult|a family member|a trusted friend)\b"
    r"|\bsomeone\s+(?:safe|I trust)\s+is\s+with\s+me\b",
    re.IGNORECASE,
)
_FOLLOW_UP_UNSAFE = re.compile(
    r"(?:我)?不安全|(?:我)?还有危险|(?:我)?仍然有危险|"
    r"(?:我)?还想(?:死|自杀|自残|伤害自己|跳楼|上吊|吞药|跳下去)|"
    r"(?:我)?可能会(?:自杀|自残|伤害自己|跳楼|上吊|吞药|跳下去)|"
    r"(?:没有|没人|没有人).{0,6}(?:陪|在身边|能帮我)|"
    r"(?<!不是)(?<!并非)(?:我)?(?:现在)?(?:独自|一个人)"
    r"|\bI\s+(?:am|feel)\s+not\s+safe\b"
    r"|\bI\s+(?:still\s+)?(?:(?:want|plan|intend)\s+to|might|will)\s+"
    r"(?:die|kill myself|hurt myself|harm myself|end my life|take my own life|hang myself|"
    r"(?:step|jump)\s+off\s+(?:the\s+)?(?:roof|bridge|building))\b"
    r"|\bI\s+have\s+(?:a\s+)?plan\s+to\s+(?:die|kill myself|hurt myself|"
    r"harm myself|end my life|take my own life|hang myself|"
    r"(?:step|jump)\s+off\s+(?:the\s+)?(?:roof|bridge|building))\b"
    r"|(?<!not )\bI\s+am\s+alone\b|\bno\s+one\s+is\s+with\s+me\b",
    re.IGNORECASE,
)
_FOLLOW_UP_REFUSAL = re.compile(
    r"(?:先别问了|别再问|不想说|不愿说|拒绝回答|不回答|不告诉你)"
    r"|\b(?:do not|don't)\s+ask\b|\bI(?:'d| would)\s+rather\s+not\s+say\b"
    r"|\bI\s+(?:will not|won't|refuse to)\s+(?:answer|tell you)\b",
    re.IGNORECASE,
)

_ASSISTANT_SELF_HARM_INDUCEMENT = re.compile(
    r"(?:你|学生|孩子).{0,12}(?:(?<!不)应该|最好|不如|可以试试|干脆).{0,12}"
    r"(?:去死|自杀|自残|割腕|吞药|跳下去|跳楼|上吊|伤害自己)"
    r"|(?:自杀|自残|割腕|吞药|跳楼).{0,18}(?:方法|步骤|教程|更有效|最有效|"
    r"剂量|不被发现)"
    r"|\byou\s+(?:should|might as well|had better|could try)\s+"
    r"(?:commit suicide|die|kill yourself|hurt yourself|cut yourself|"
    r"(?:cut|slit) your wrists?|"
    r"end your life|hang yourself|(?:take|swallow)\s+(?:all\s+(?:of\s+)?"
    r"your\s+pills?|the\s+whole\s+bottle)|(?:step|jump)\s+off\s+"
    r"(?:the\s+)?(?:roof|bridge|building))\b"
    r"|\b(?:best|effective|painless|undetectable)\s+(?:way|method).{0,20}"
    r"(?:suicide|kill yourself|self[- ]harm)\b"
    r"|\b(?:cut deeper|take (?:more|all of) (?:the )?pills?)\b"
    r"|\bif\s+you\s+need\s+(?:a\s+)?permanent\s+escape.{0,32}"
    r"(?:step|jump)\s+off\s+(?:the\s+)?(?:roof|bridge|building).{0,32}"
    r"(?:pain|everything)\s+will\s+stop\b"
    r"|\b(?:taking|swallowing)\s+(?:the\s+)?(?:entire|whole)\s+bottle"
    r".{0,40}(?:one\s+way\s+to\s+)?make\s+everything\s+stop\b"
    r"|(?:把|将).{0,8}(?:手里(?:的)?|全部|整瓶)?(?:药|药片).{0,8}"
    r"(?:一次)?(?:吃|吞)(?:光|完|下).{0,24}(?:再也)?不用(?:再)?痛苦",
    re.IGNORECASE,
)
_ASSISTANT_SELF_HARM_INSTRUCTION = re.compile(
    r"(?:^|[.!?。！？]\s+)(?:"
    r"here(?:'s|\s+is|\s+are)\s+(?:(?:the\s+)?steps?\s+to|how\s+to)\s+"
    r"(?:commit\s+suicide|kill\s+yourself|end\s+your\s+life)|"
    r"to\s+(?:commit\s+suicide|kill\s+yourself|end\s+your\s+life),?\s+first\b|"
    r"I\s+can\s+explain\s+(?:a\s+)?(?:suicide|self[- ]harm)\s+method\s+"
    r"step\s+by\s+step|use\s+(?:a\s+)?rope\s+to\s+hang\s+yourself|"
    r"you\s+can\s+end\s+your\s+life\s+by\s+(?:jumping|stepping)\s+off\s+"
    r"(?:a|the)\s+(?:roof|bridge|building)|(?<!不)(?<!不能)(?<!不会)"
    r"教你如何(?:结束生命|自杀|自残))",
    re.IGNORECASE,
)
_ASSISTANT_DIRECT_SELF_HARM_IMPERATIVE = re.compile(
    r"^\s*(?:kill yourself|go die|end your life|hang yourself|"
    r"(?:go\s+)?(?:jump|step)\s+off\s+(?:a|the)?\s*(?:roof|bridge|building)|"
    r"(?:cut|slit)\s+(?:your\s+)?wrists?|(?:take|swallow)\s+(?:all\s+(?:of\s+)?"
    r"your\s+pills?|the\s+whole\s+bottle)|why\s+(?:do\s+not|don't)\s+"
    r"you\s+(?:kill|hurt)\s+yourself|去死|自杀吧|结束生命吧|"
    r"跳下去吧|跳楼吧|上吊吧|割腕吧|"
    r"从(?:桥上|天台|楼顶)跳下去吧?|"
    r"把?(?:整瓶|全部)(?:药|药片)(?:吞下去|吞掉)吧?|"
    r"(?:吞掉|吞下|吃掉)(?:整瓶|全部)(?:药|药片)吧?)[.!?。！？]*\s*$"
    r"|(?:^|[.!?。！？]\s+)(?:kill yourself|go die|end your life|hang yourself|"
    r"去死|自杀吧|结束生命吧|跳下去吧|跳楼吧|上吊吧|割腕吧)"
    r"[.!?。！？]*(?:\s|$)",
    re.IGNORECASE,
)
_ASSISTANT_CONCEALED_OVERDOSE_INDUCEMENT = re.compile(
    r"\b(?:take|swallow)\s+(?:the\s+)?(?:whole\s+bottle|all\s+(?:of\s+)?"
    r"(?:the\s+)?pills?).{0,36}\b(?:nobody|no\s+one).{0,16}"
    r"(?:needs?|has)\s+to\s+know\b"
    r"|(?:整瓶|全部).{0,6}(?:药|药片).{0,20}(?:没人|不要让人|别让人)"
    r".{0,12}(?:知道|发现)",
    re.IGNORECASE,
)
_ASSISTANT_ABUSIVE_INSULT = re.compile(
    r"(?:你|这个学生)(?:真|太|就是|简直)?(?:蠢|笨|废物|白痴|没用|恶心)"
    r"|(?:蠢货|笨蛋|白痴|废物)，?你"
    r"|\byou(?:'re| are)\s+(?:an?\s+)?(?:idiot|stupid|worthless|useless|disgusting|moron)\b"
    r"|\byou\s+are\s+hopeless.{0,32}\bnobody\s+would\s+miss\s+you\b",
    re.IGNORECASE,
)
_ASSISTANT_MINOR_SEXUAL = re.compile(
    r"(?:未成年|儿童|小学生|初中生|[1-9]|1[0-7]岁).{0,24}"
    r"(?:裸照|裸体|色情|性行为|性描写|成人视频)"
    r"|(?:裸照|裸体|色情|性行为|性描写).{0,24}(?:未成年|儿童|小学生|初中生)"
    r"|\b(?:child|minor|underage|[1-9]|1[0-7][ -]year[ -]old).{0,36}"
    r"(?:nude|naked|sexual|explicit|porn|sex scene)\b",
    re.IGNORECASE,
)
_ASSISTANT_MINOR_SEXUAL_REFUSAL = re.compile(
    r"(?:不能|不会|拒绝).{0,24}(?:制作|寻找|描述|提供).{0,24}"
    r"(?:未成年|儿童).{0,16}(?:性内容|色情|裸照)"
    r"|\b(?:cannot|can't|will not|won't|refuse to).{0,28}"
    r"(?:create|find|describe|provide).{0,28}(?:sexual content|porn|nudes?)"
    r".{0,20}(?:minor|child|underage)\b",
    re.IGNORECASE,
)
_ASSISTANT_DANGEROUS_MEDICAL = re.compile(
    r"(?:你|患者).{0,10}(?:直接|马上|应该|可以).{0,8}"
    r"(?:停药|加倍剂量|自行加药|混合服药|不用看医生|不要就医)"
    r"|(?:不用|不要).{0,8}(?:联系急救|去医院|看医生)"
    r"|\byou\s+(?:should|can)\s+(?:stop|double|increase|change).{0,18}"
    r"(?:medication|medicine|dose|dosage)\b"
    r"|\b(?:stop taking|double|increase|change)\s+(?:your\s+)?"
    r"(?:medication|medicine|dose|dosage)\b"
    r"|\b(?:do not|don't)\s+(?:call emergency services|seek medical care|see a doctor)\b",
    re.IGNORECASE,
)


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip()


def classify_safety_follow_up(text: str) -> str:
    """Classify a safeguarding follow-up without inferring reassurance.

    ``explicitly_safe`` requires both an unambiguous current-safety statement
    and a named trusted-support condition.  Ambiguity, refusal, silence, and a
    safety statement without support remain ``unknown``; any unsafe/alone cue
    wins over a contradictory safety assertion.
    """

    value = _normalized_text(text)
    if not value:
        return "unknown"
    unsafe_view = re.sub(
        r"(?:我不是一个人|我并非一个人|I(?:'m| am) not alone)",
        "",
        value,
        flags=re.IGNORECASE,
    )
    risk_view = _EXPLICIT_SELF_HARM_DENIAL.sub("", unsafe_view)
    if _FIRST_PERSON_SELF_HARM.search(risk_view) or _FOLLOW_UP_UNSAFE.search(
        unsafe_view
    ):
        return "still_unsafe"
    if _FOLLOW_UP_REFUSAL.search(value):
        return "unknown"
    if _FOLLOW_UP_EXPLICIT_SAFE.search(value) and _FOLLOW_UP_TRUSTED_SUPPORT.search(
        value
    ):
        return "explicitly_safe"
    return "unknown"


def classify_assistant_output_safety(text: str) -> dict[str, Any] | None:
    """Return a content-free receipt when generated learner-visible text is unsafe."""

    value = _normalized_text(text)
    if not value:
        return None
    category: str | None = None
    if (
        _ASSISTANT_SELF_HARM_INDUCEMENT.search(value)
        or _ASSISTANT_SELF_HARM_INSTRUCTION.search(value)
        or _ASSISTANT_DIRECT_SELF_HARM_IMPERATIVE.search(value)
        or _ASSISTANT_CONCEALED_OVERDOSE_INDUCEMENT.search(value)
    ):
        category = "self_harm_inducement"
    elif _ASSISTANT_ABUSIVE_INSULT.search(value):
        category = "abusive_insult"
    elif _ASSISTANT_MINOR_SEXUAL.search(
        value
    ) and not _ASSISTANT_MINOR_SEXUAL_REFUSAL.search(value):
        category = "minor_sexual_content"
    elif _ASSISTANT_DANGEROUS_MEDICAL.search(value):
        category = "dangerous_medical_instruction"
    if category is None:
        return None
    return {
        "schema": ASSISTANT_OUTPUT_SAFETY_SCHEMA,
        "category": category,
        "severity": "high",
        "output_sha256": sha256(value.encode("utf-8")).hexdigest(),
        "output_text_persisted": False,
        "visible_bytes_before_validation": 0,
        "policy": "block_before_visibility_or_persistence",
        "policy_version": "student_safety_output_boundary_v1",
    }


def fixed_generation_safety_response(*, chinese: bool) -> str:
    """Return the fixed safe replacement used after a generated-output block."""

    return (
        "我不能提供可能伤害学习者的内容。我们先暂停这个方向；如涉及即时安全或医疗风险，请联系可信任的成年人以及当地专业或紧急服务。可以改为讨论安全的一般原理。"
        if chinese
        else "I cannot provide content that could harm a learner. We will pause this direction; if there is an immediate safety or medical risk, contact a trusted adult and local professional or emergency support. We can discuss safe general principles instead."
    )


def fixed_safety_follow_up_response(status: str, *, chinese: bool) -> str:
    """Return the fixed tri-state safeguarding follow-up without inference."""

    if status not in SAFETY_FOLLOW_UP_STATUSES:
        raise ValueError("safety follow-up status is invalid")
    if status == "still_unsafe":
        return (
            "我们先不回到学业任务。若你仍不安全，请立即联系当地急救服务或可信任的成年人并让对方陪着你。你现在是否有可信任的人陪着？"
            if chinese
            else "We will not return to the lesson yet. If you remain unsafe, contact local emergency services or a trusted adult and ask them to stay with you. Is a trusted person with you now?"
        )
    if status == "explicitly_safe":
        return (
            "谢谢你明确告诉我你现在安全且有可信任的人陪着。请继续和对方待在一起；你可以回复“恢复课程”。"
            if chinese
            else "Thank you for clearly confirming that you are safe and with trusted support. Stay with them; you may reply “resume lesson”."
        )
    return (
        "我还不能确认你现在安全。请明确告诉我：你此刻是否安全，以及是否有可信任的成年人或支持人员陪着你。"
        if chinese
        else "I still cannot confirm that you are safe. Please state clearly whether you are safe now and whether a trusted adult or support person is with you."
    )


def _response(category: str, *, chinese: bool, urgent: bool) -> str:
    if category == "self_harm":
        if chinese:
            urgency = (
                "如果你现在可能马上伤害自己，请立即联系当地急救服务，或让身边可信任的成年人陪着你。"
                if urgent
                else "请先联系一位可信任的成年人、学校心理老师或当地危机援助服务，不要独自扛着。"
            )
            return (
                "我很在意你刚才说的事，先暂停课程；你的安全比答题重要。"
                + urgency
                + "如果可以，把可能伤害你的物品先放远并去到有人在的地方。你现在是否处在马上会伤害自己的危险中？"
            )
        urgency = (
            "If you may act now, contact local emergency services or ask a trusted adult to stay with you immediately."
            if urgent
            else "Please contact a trusted adult, school counselor, or local crisis service and do not handle this alone."
        )
        return (
            "I am glad you told me. We are pausing the lesson because your safety matters more than the task. "
            + urgency
            + " Move away from anything you could use to hurt yourself and go where another person is present if you can. Are you in immediate danger of hurting yourself right now?"
        )
    if category in {"abuse_disclosure", "bullying_disclosure"}:
        if chinese:
            return (
                "这不是你的错，我们先暂停课程，也不把这段话当作学习成绩。"
                "请尽快告诉一位你信任且不会伤害你的成年人或学校保护人员；如果眼下不安全，就离开现场并联系当地紧急服务。你现在所在的地方安全吗？"
            )
        return (
            "This is not your fault. We are pausing the lesson and will not treat this as learning evidence. "
            "Tell a trusted safe adult or school safeguarding contact as soon as you can; if you are unsafe now, leave the area and contact local emergency services. Are you somewhere safe right now?"
        )
    if category == "urgent_medical":
        return (
            "这可能需要立即的专业帮助。请暂停课程并联系当地急救服务或身边可信任的成年人；我不能通过教学对话判断或处理急症。现在有人能马上陪你寻求帮助吗？"
            if chinese
            else "This may need immediate professional help. Pause the lesson and contact local emergency services or a trusted adult; I cannot assess or treat an emergency in a teaching chat. Is someone available to help you seek care now?"
        )
    if category == "minor_sexual_content":
        return (
            "我不能帮助制作、寻找或描述涉及未成年人的性内容。若这是现实中的接触、威胁或影像，请停止转发并告诉可信任的成年人或当地保护机构。你是想学习安全与求助知识吗？"
            if chinese
            else "I cannot help create, find, or describe sexual content involving minors. If this concerns real contact, threats, or images, do not forward them and tell a trusted adult or local safeguarding service. Are you asking for safety and reporting guidance?"
        )
    if category == "high_risk_advice":
        return (
            "这涉及医疗或法律上的高风险决定，我不能替专业人员给出个案指令。我们可以学习一般原理，但具体行动应由合格医生或律师结合完整情况确认。你想先了解哪条一般原理？"
            if chinese
            else "This is a high-risk medical or legal decision, so I cannot give case-specific instructions. We can study general principles, but a qualified clinician or lawyer should assess the full situation. Which general principle would you like to understand first?"
        )
    if category == "academic_integrity":
        return (
            "我不能代考、伪造数据或直接交付可冒充你本人完成的作业；但可以讲清方法、示范一个不同例题，再陪你完成当前任务的一小步。你希望先看方法还是一个相似例子？"
            if chinese
            else "I cannot take an exam, fabricate data, or produce work to pass off as your own. I can explain the method, model a different example, and coach one step of your task. Would you prefer the method or a similar example first?"
        )
    return (
        "这种说法会伤害一个群体，也不能作为学习结论。我们可以改用可核验事实讨论具体行为与证据，而不把特征归咎于整个群体。你愿意把问题改成一个可核验的具体判断吗？"
        if chinese
        else "That claim targets a whole group and is not an acceptable learning conclusion. We can discuss specific behavior and evidence without blaming an identity group. Can you restate it as one verifiable claim?"
    )


def classify_learner_safety(
    text: str,
    student_profile: Mapping[str, Any] | None = None,
    *,
    source_kind: str = "learner_text",
    objective_educational_context: bool = False,
) -> dict[str, Any] | None:
    """Return a deterministic safety contract, or ``None`` for normal teaching."""

    value = _normalized_text(text)
    if not value:
        return None
    if source_kind not in {
        "learner_text",
        "learner_ocr",
        "chat_user",
        "teach_goal",
        "learner_profile",
        "teaching_resource",
    }:
        raise ValueError("learner safety source_kind is unsupported")
    objective_context = bool(
        objective_educational_context and _OBJECTIVE_SAFETY_EDUCATION.search(value)
    )
    personal_risk_view = _EXPLICIT_SELF_HARM_DENIAL.sub("", value)
    category: str | None = None
    severity = "elevated"
    self_harm_matches = list(_FIRST_PERSON_SELF_HARM.finditer(personal_risk_view))

    def reported_context(match: re.Match[str]) -> bool:
        # A literary/news marker elsewhere in the same message must never mask
        # a learner's own current disclosure.  Treat a match as reported only
        # when a concrete attribution is immediately before that exact match,
        # with no first-person/contrast pivot in between.
        prefix_start = max(0, match.start() - 120)
        prefix = personal_risk_view[prefix_start : match.start()]
        contexts = list(_REPORTED_SELF_HARM_EXCLUSION.finditer(prefix))
        if not contexts:
            return False
        gap = prefix[contexts[-1].end() :]
        attributed_tail = gap + match.group(0)
        return (
            len(gap) <= 48 and _PERSONAL_CONTEXT_PIVOT.search(attributed_tail) is None
        )

    personal_self_harm = any(not reported_context(match) for match in self_harm_matches)
    if personal_self_harm:
        category = "self_harm"
        severity = "urgent" if _IMMINENCE.search(value) else "high"
    elif _RESOURCE_HARMFUL_SELF_HARM_INSTRUCTION.search(value):
        category = "self_harm"
        severity = "high"
    elif _ABUSE_DISCLOSURE.search(value) and not objective_context:
        category = "abuse_disclosure"
        severity = "high"
    elif _BULLYING_DISCLOSURE.search(value) and not objective_context:
        category = "bullying_disclosure"
        severity = "high"
    elif _MINOR_SEXUAL_REQUEST.search(value) and not objective_context:
        category = "minor_sexual_content"
        severity = "high"
    elif _URGENT_MEDICAL.search(value) and not objective_context:
        category = "urgent_medical"
        severity = "urgent"
    elif _HIGH_RISK_ADVICE.search(value) and not objective_context:
        category = "high_risk_advice"
    elif _CHEATING_REQUEST.search(value):
        category = "academic_integrity"
    elif _TARGETED_HATE.search(value) and not _CRITICAL_HATE_EXCLUSION.search(value):
        category = "targeted_hate"
    if category is None:
        return None
    profile = student_profile if isinstance(student_profile, Mapping) else {}
    minor_status = "unknown"
    learner_level = str(profile.get("learner_level", "")).casefold()
    age = profile.get("age")
    if any(
        term in learner_level for term in ("小学", "初中", "中学", "child", "minor")
    ) or (isinstance(age, int) and not isinstance(age, bool) and 0 < age < 18):
        minor_status = "likely_minor_profile"
    policy = (
        "pause_and_escalate"
        if category
        in {
            "self_harm",
            "abuse_disclosure",
            "bullying_disclosure",
            "minor_sexual_content",
            "urgent_medical",
        }
        else "bounded_general_information"
        if category == "high_risk_advice"
        else "integrity_preserving_scaffold"
        if category == "academic_integrity"
        else "deescalate_and_reframe"
    )
    return {
        "schema": LEARNER_SAFETY_CONTRACT_SCHEMA,
        "category": category,
        "severity": severity,
        "policy": policy,
        "response": _response(
            category, chinese=bool(_CJK.search(value)), urgent=severity == "urgent"
        ),
        "learner_text_sha256": sha256(value.encode("utf-8")).hexdigest(),
        "learner_text_persisted": False,
        "input_origin": source_kind,
        "objective_educational_context": objective_context,
        "minor_status": minor_status,
        "mastery_evidence": False,
        "hold_lesson_phase": True,
        "remote_model_required": False,
        "requires_human_review": category
        not in {"academic_integrity", "targeted_hate"},
        # No safeguarding case queue or staffed review integration exists in
        # this runtime. ``requires_human_review`` is a recommendation only.
        "human_escalation_status": "unavailable",
        "human_escalation_triggered": False,
        "diagnosis_or_professional_advice_provided": False,
        "policy_version": "student_safety_boundary_v2",
        "jurisdiction_specific_instruction_provided": False,
        "emergency_resource_localization_required": severity == "urgent",
        "emergency_resource_localization_status": "unavailable",
        "emergency_resource_guidance_scope": "generic_local_services_only",
        "safety_follow_up_status": "unknown",
    }


def classify_learner_safety_fields(
    value: Any,
    student_profile: Mapping[str, Any] | None = None,
    *,
    source_kind: str,
    objective_educational_context: bool = False,
    maximum_nodes: int = 4_096,
) -> dict[str, Any] | None:
    """Scan bounded nested learner-controlled fields one string at a time."""

    stack = [value]
    observed = 0
    while stack:
        observed += 1
        if observed > maximum_nodes:
            raise ValueError("learner safety field scan exceeds its node bound")
        current = stack.pop()
        if isinstance(current, str):
            contract = classify_learner_safety(
                current,
                student_profile,
                source_kind=source_kind,
                objective_educational_context=objective_educational_context,
            )
            if contract is not None:
                return contract
        elif isinstance(current, Mapping):
            stack.extend(reversed(tuple(current.values())))
        elif isinstance(current, (list, tuple)):
            stack.extend(reversed(current))
    return None


__all__ = [
    "ASSISTANT_OUTPUT_SAFETY_SCHEMA",
    "LEARNER_SAFETY_CONTRACT_SCHEMA",
    "SAFETY_FOLLOW_UP_STATUSES",
    "classify_assistant_output_safety",
    "classify_learner_safety",
    "classify_learner_safety_fields",
    "classify_safety_follow_up",
    "fixed_generation_safety_response",
    "fixed_safety_follow_up_response",
]
