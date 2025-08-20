from lighteval.metrics.metrics_sample import (
    PassAtK,
)
from lighteval.utils.language import Language
from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    IndicesExtractionConfig,
    LatexExtractionConfig,
    compare_gold_target,
    extract_target_from_pred,
    get_extraction_regexes,
    multilingual_extractive_match_metric,
)
import numpy as np
from lighteval.metrics.metrics_sample import ExactMatches
from typing import Callable
import os
import numpy as np
from lighteval.metrics.normalizations import helm_normalizer
from lighteval.metrics.utils.metric_utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetric,
)


# Metric for GPQA-Diamond-fr task
gpqa_instruct_pass_fr_at_1_1n = SampleLevelMetric(
    metric_name="gpqa_pass_fr@1:1_samples",
    sample_level_fn=PassAtK(
        k=1,
        n=1,
        sample_scoring_function=lambda pred, ref, doc: multilingual_extractive_match_metric(
            language=Language.FRENCH,
            gold_extraction_target=[
                IndicesExtractionConfig(prefix_for_extraction="NativeLetters")
            ],
            pred_extraction_target=[
                IndicesExtractionConfig(prefix_for_extraction="NativeLetters")
            ],
            precision=6,
        ).sample_level_fn(
            [ref], [pred], doc
        ),
    ).compute,
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.REASONING,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

math_fr_pass_at_1_1n = SampleLevelMetric(
    metric_name="math_fr_pass@1:1_samples",
    sample_level_fn=PassAtK(
        k=1,
        n=1,
        strip_strings=True,
        # Extracting mathematical expressions and latex expressions
        normalize_gold=lambda k: extract_target_from_pred(
            k,
            get_extraction_regexes(
                formatted_doc=None,
                target_types=[ExprExtractionConfig(), LatexExtractionConfig()],
                language=Language.FRENCH,
            ),
        ),
        # Extracting mathematical expressions and latex expressions
        normalize_pred=lambda k: extract_target_from_pred(
            k,
            get_extraction_regexes(
                formatted_doc=None,
                target_types=[ExprExtractionConfig(), LatexExtractionConfig()],
                language=Language.FRENCH,
            ),
        ),
        # Uses sympy for comparison
        sample_scoring_function=compare_gold_target,
    ).compute,
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.REASONING,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


from sympy import (
    And,
    Basic,
    E,
    Eq,
    FiniteSet,
    Float,
    GreaterThan,
    Interval,
    LessThan,
    MatrixBase,
    MatrixExpr,
    Mul,
    Number,
    Rational,
    Set,
    StrictGreaterThan,
    StrictLessThan,
    Symbol,
    Tuple,
    default_sort_key,
    ordered,
    simplify,
)
from sympy.core.function import UndefinedFunction
from sympy.core.relational import Relational
from lighteval.utils.timeout import timeout
from itertools import product
from sympy.parsing.latex import parse_latex
import re

def normalize_expression(expr: str) -> str:
    """
    Normalizes a LaTeX or text expression for SymPy comparison:
    - converts decimal commas to dots
    - removes LaTeX spaces (\, and ~)
    - removes common SI units (\text{V}, \text{C}, \text{J}, etc.)
    - standardizes certain symbols (R–OH -> R-OH, apostrophes)
    """

    if not isinstance(expr, str):
        return expr

    expr = expr.replace(r"\,", "")
    expr = expr.replace("~", "")

    expr = expr.replace("{,}", ".")
    expr = expr.replace(",", ".")

    expr = expr.replace("’", "'")

    expr = expr.replace("–", "-")

    units = [
        "V", "A", "Ω", "ohm", "C", "J", "W", "F", "H", "mol", "s", "m", "g", "kg",
        "N", "Pa", "Hz", "K", "rad", "cd", "lx", "T", "Wb"
    ]

    for u in units:
        expr = re.sub(rf"\\text{{\s*{u}\s*}}", "", expr)   # \text{V}
        expr = re.sub(rf"\\mathrm{{\s*{u}\s*}}", "", expr) # \mathrm{V}
        expr = re.sub(rf"(?<![A-Za-z]){u}(?![A-Za-z])", "", expr) 

    expr = expr.strip()

    return expr


def safe_parse_latex(s: str):
    '''
    Extract all contents in \boxed{} and convert it to a datastructure if it's a tuple, a list ...
    '''
    s = s.strip()
    if re.match(r"^[\(\{\[].*[\)\}\]]$", s):

        try:
            s_clean = s.replace(r"\left", "").replace(r"\right", "")
            s_clean = s_clean.replace("{", "[").replace("}", "]")
            return eval(s_clean) 
        except Exception:
            return normalize_expression(s)
    try:
        return parse_latex(normalize_expression(s))
    except Exception:
        return s


def compare_gold_target2(
    gold: list[Basic | MatrixBase | str] | Basic | MatrixBase | str,
    target: list[Basic | MatrixBase | str] | Basic | MatrixBase | str,
    precision: int = 6,
    strict: bool = True,
    timeout_seconds: int = 3,
) -> bool:
    @timeout(timeout_seconds=timeout_seconds)
    def compare_single_extraction(gold: Basic | MatrixBase | str, target: Basic | MatrixBase | str) -> bool:

        
        expr1 = safe_parse_latex(gold)
        expr2 = safe_parse_latex(target)
        #print(f"gold: {expr1}, target: {expr2}")
        if isinstance(expr1, (Basic, MatrixBase)) and isinstance(expr2, (Basic, MatrixBase)):
            return expr1.equals(expr2)

        # For intervals, vectors
        elif isinstance(expr1, (list, tuple)) and isinstance(expr2, (list, tuple)):
            return list(expr1) == list(expr2)

        elif isinstance(gold, str) and isinstance(target, str):
            # We just do string comparison for everything else
            gold = gold.strip()
            target = target.strip()

            # Ensure it's both not empty and equal
            return len(gold) > 0 and len(target) > 0 and gold == target
        
        return False

    def compare_single_extraction_wrapper(g, t):
        try:
            return compare_single_extraction(g, t)
        except Exception as e:  # noqa: E722
            print(f"Error when evaluating:{e}")
            return False

    return any(compare_single_extraction_wrapper(g, t) for g, t in product(gold, target))

    
def extract_boxed(text: str) -> list[str]:
    results = []
    i = 0
    while True:
        start = text.find(r"\boxed{", i)
        if start == -1:
            break
        # position après "\boxed{"
        j = start + len(r"\boxed{")
        depth = 1
        content = []
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            content.append(text[j])
            j += 1
        results.append("".join(content)) 
        i = j + 1
    return results

def extract_gold(text: str) -> list[str]:
    return [text]


kholle_pass_at_1_1n = SampleLevelMetric(
    metric_name="kholle_fr_pass@1:1_samples",
    sample_level_fn=PassAtK(
        k=1,
        n=1,
        strip_strings=True,
        normalize_gold=lambda k: extract_gold(k) ,
        # Extracting mathematical expressions and latex expressions
        normalize_pred=lambda k: extract_boxed(k),
        # Uses sympy for comparison
        sample_scoring_function=compare_gold_target2,
    ).compute,
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.REASONING,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

class ExactMatchesThinking(ExactMatches):
    """
    A class to compute exact matches for reasoning tasks.
    This class extends the ExactMatches metric to handle reasoning-specific requirements.
    """

    def __init__(
        self,
        aggregation_function: Callable[[list[float]], float] = max,
        normalize_gold: Callable[[str], str] | None = None,
        normalize_pred: Callable[[str], str] | None = None,
        strip_strings: bool = False,
        type_exact_match: str = "full",
        answer_token: str = os.environ.get(
            "answer_token", ""
        ),  # for reasoning tasks we need to specify the answer token like <answer> or end of thinking token "</thinking>" if no asnwer token is outputted
    ):

        super().__init__(
            aggregation_function=aggregation_function,
            normalize_gold=normalize_gold,
            normalize_pred=normalize_pred,
            strip_strings=strip_strings,
            type_exact_match=type_exact_match,
        )
        self.answer_token = answer_token

    def compute_one_item(
        self,
        gold: str,
        pred: str,
    ) -> float:

        # extract the answer afte the answer token if it exists
        if self.answer_token:
            if self.answer_token in pred:
                pred = pred.split(self.answer_token, 1)[1]

        return super().compute_one_item(gold, pred)


class MetricsThinking:
    exact_match = SampleLevelMetric(
        metric_name="em",
        sample_level_fn=ExactMatchesThinking(strip_strings=True).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )

    quasi_exact_match = SampleLevelMetric(
        metric_name="qem",
        sample_level_fn=ExactMatchesThinking(
            normalize_gold=helm_normalizer,
            normalize_pred=helm_normalizer,
            strip_strings=True,
        ).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )

    prefix_exact_match = SampleLevelMetric(
        metric_name="pem",
        sample_level_fn=ExactMatchesThinking(
            strip_strings=True, type_exact_match="prefix"
        ).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )

    prefix_quasi_exact_match = SampleLevelMetric(
        metric_name="pqem",
        sample_level_fn=ExactMatchesThinking(
            normalize_gold=helm_normalizer,
            normalize_pred=helm_normalizer,
            type_exact_match="prefix",
        ).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
