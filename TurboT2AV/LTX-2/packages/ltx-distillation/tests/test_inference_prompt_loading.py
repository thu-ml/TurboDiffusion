from pathlib import Path

from ltx_distillation.tools.run_av_inference_eval import _load_prompts


def test_load_prompts_treats_headerless_csv_as_one_prompt_per_line(tmp_path: Path) -> None:
    prompts_file = tmp_path / "prompts.csv"
    prompts_file.write_text("first prompt, with a comma\nsecond prompt\n", encoding="utf-8")

    assert _load_prompts(str(prompts_file), limit=None) == [
        "first prompt, with a comma",
        "second prompt",
    ]


def test_load_prompts_reads_named_csv_column_and_applies_limit(tmp_path: Path) -> None:
    prompts_file = tmp_path / "prompts.csv"
    prompts_file.write_text('video_id,prompt\n0,"first prompt, with a comma"\n1,second prompt\n', encoding="utf-8")

    assert _load_prompts(str(prompts_file), limit=1) == ["first prompt, with a comma"]
