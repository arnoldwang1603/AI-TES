"""Export the registered material-change maintenance column, without fitting.

The statements are protocol annotations, not measured transfer performance.
They describe output/reference parameters, not whether a trained GRU can be
used on a new material without training-data scaling or retraining.
"""
from pathlib import Path

from paper_report import read_json

HERE = Path(__file__).resolve().parent
REFIT_COLUMN = "换材料时每种输出形式要重拟合什么"
REFIT_SOURCE = "batches.P3.re_fit_column"
REFIT_SCOPE = (
    "本列来自预注册，仅说明输出参照的附加拟合参数。direct 的 none 不代表"
    "换材料后训练数据缩放器或 GRU 权重无需重新拟合，也不代表材料迁移已验证。"
    "physics_input_refit 单独列出物理输入需要更新的物性。"
)
CHANNELS = ("T_inner", "T_outer", "T_avg")
FORMS = ("direct", "anchor", "EMA", "ratio", "physics residual")


def read_refit_rules(prereg=None):
    """Read exact parameter descriptions from the frozen protocol."""
    if prereg is None:
        prereg = HERE / "prereg" / "round10_v4.json"
    p = prereg if isinstance(prereg, dict) else read_json(prereg)
    text = p["batches"]["P3"]["re_fit_column"]
    rules = {}
    for cell in text.split(";"):
        name, separator, value = cell.partition(":")
        name, value = name.strip(), value.strip()
        if not separator or not value or name in rules:
            raise RuntimeError("Invalid registered material refit column: " + cell)
        rules[name] = value
    if set(rules) != set(FORMS):
        raise RuntimeError("Material refit column must define the five registered output forms")
    return rules


def channel_refit_metadata(config, prereg=None):
    """Classify actual heads per channel; shared physical inputs are separate."""
    rules = read_refit_rules(prereg)
    other = config.get("other_ch_mode", "abs")
    physics = config.get("phys_mode", "")
    outer = config.get("touter_mode", "abs")
    inner = config.get("tinner_mode", "anchor")
    if other not in ("abs", "ratio_input", "anchor_initial"):
        raise RuntimeError("Unregistered Round10 output form: " + str(other))
    if physics not in ("", "residual", "input", "input_residual"):
        raise RuntimeError("Unregistered Round10 physics mode: " + str(physics))
    ratio = {"R1": {1}, "R2": {1, 2}, "R3": {0, 1, 2}}.get(config.get("ratio_channels", "R3"))
    if other == "ratio_input" and ratio is None:
        raise RuntimeError("Unregistered Round10 ratio channel set")
    result = {}
    for index, channel in enumerate(CHANNELS):
        if physics in ("residual", "input_residual") and index in (1, 2):
            form = "physics residual"
        elif other == "ratio_input" and index in ratio:
            form = "ratio"
        elif index == 0:
            if inner == "anchor":
                form = "anchor"
            elif inner in ("arfed", "abs"):
                form = "direct"
            else:
                raise RuntimeError("Unregistered Round10 T_inner output form: " + str(inner))
        elif index == 1 and str(outer).startswith("anchor_ema"):
            form = "EMA"
        elif index == 1 and outer == "anchor_input":
            form = "anchor"
        elif other == "anchor_initial":
            form = "anchor"
        else:
            if index == 1 and outer != "abs":
                raise RuntimeError("Unregistered Round10 T_outer output form: " + str(outer))
            form = "direct"
        result[channel] = dict(
            output_form=form,
            physics_input_refit=rules["physics residual"] if physics in ("input", "input_residual") else rules["direct"],
            refit_rule_source=REFIT_SOURCE,
        )
        result[channel][REFIT_COLUMN] = rules[form]
    return result


def write_refit_guide(out, prereg=None):
    """Keep all five statements visible, including the unlaunched EMA form."""
    import pandas as pd
    from analysis_common import write_table
    rules = read_refit_rules(prereg)
    rows = [{"output_form": form, REFIT_COLUMN: rules[form],
             "refit_rule_source": REFIT_SOURCE,
             "status": "protocol annotation only; no new-material validation"}
            for form in FORMS]
    write_table(pd.DataFrame(rows), out, "material_refit_guide", REFIT_SCOPE)
