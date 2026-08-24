"""Canonical method colors shared by all result plots."""

METHOD_COLOR = {
    "bad_teacher": "#1f77b4",
    "delete": "#ff7f0e",
    "gradient_ascent": "#2ca02c",
    "random_label": "#d62728",
    "salun": "#9467bd",
    "retrained": "#8c564b",
    "finetune": "#e377c2",
    "boundary_shrink": "#7f7f7f",
    "l2ul_adv": "#bcbd22",
    "scrub": "#17becf",
    "neggrad_plus": "#000000",
}


def get_method_color(method_key: str) -> str:
    return METHOD_COLOR.get(method_key, "#555555")
