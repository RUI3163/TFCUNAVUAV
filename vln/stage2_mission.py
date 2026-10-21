"""Stage 2: Mission understanding. Qwen2.5-3B-Instruct -> Pydantic-validated JSON spec.

Runs ONCE at start (greedy decoding), emits no control output, and is unloaded from VRAM
before any other model is loaded. There is no silent fallback: if validation fails after
`max_retries`, MissionParseError is raised.
"""
from __future__ import annotations

import json
import logging
import re
from typing import List, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from .config import MissionLLMCfg
from .gpu import bnb_4bit_config, free_vram, log_vram

log = logging.getLogger(__name__)

Color = Literal["red", "blue", "green", "white", "black", "silver", "yellow", "orange"]


class TargetSpec(BaseModel):
    category: Literal["passenger_car"]
    color: Color


class ReferenceSpec(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_]{1,16}$")


class RegionSpec(BaseModel):
    area_type: Literal["parking_area"]
    near_reference: str


class MissionSpec(BaseModel):
    task: Literal["search_and_report"]
    target: TargetSpec
    search_region: RegionSpec
    reference: ReferenceSpec
    report_fields: List[Literal["east_m", "north_m", "range_m", "bearing_deg", "uncertainty_m"]] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> "MissionSpec":
        if self.search_region.near_reference != self.reference.id:
            raise ValueError("search_region.near_reference must equal reference.id")
        return self


class MissionParseError(RuntimeError):
    pass


SYSTEM_PROMPT = (
    "You convert a UAV mission request into ONE JSON object and output nothing else "
    "(no prose, no markdown). The JSON must satisfy this JSON Schema:\n{schema}\n"
    "Rules: copy the target colour and the reference-point identifier from the request; "
    "report_fields must list every quantity the request asks to report "
    "(a location relative to a reference point means east_m, north_m, range_m, bearing_deg, uncertainty_m)."
)


def _extract_json(text: str) -> str:
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON object found in model output")
    return text[a:b + 1]


def _check_against_prompt(spec: MissionSpec, prompt: str) -> None:
    low = prompt.lower()
    if spec.target.color not in low:
        raise ValueError(f"target colour '{spec.target.color}' does not appear in the request")
    if not re.search(rf"\b{re.escape(spec.reference.id)}\b", prompt):
        raise ValueError(f"reference id '{spec.reference.id}' does not appear in the request")


def parse_mission(prompt: str, cfg: MissionLLMCfg) -> MissionSpec:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Stage 2: loading %s (4-bit=%s)", cfg.model_id, cfg.load_4bit)
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    kw = dict(quantization_config=bnb_4bit_config()) if cfg.load_4bit else dict(torch_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, device_map={"": 0}, **kw)
    log_vram("stage2 loaded")
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(schema=json.dumps(MissionSpec.model_json_schema()))},
            {"role": "user", "content": prompt},
        ]
        last_err = ""
        for attempt in range(1, cfg.max_retries + 1):
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens, do_sample=False)
            raw = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            try:
                spec = MissionSpec.model_validate_json(_extract_json(raw))
                _check_against_prompt(spec, prompt)
                log.info("Stage 2: mission spec OK (attempt %d): %s", attempt, spec.model_dump_json())
                return spec
            except (ValidationError, ValueError) as exc:
                last_err = str(exc)
                log.warning("Stage 2 attempt %d invalid: %s", attempt, last_err)
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": f"Invalid: {last_err}\nReturn corrected JSON only."}]
        raise MissionParseError(f"LLM failed to produce a valid mission spec: {last_err}")
    finally:
        del model, tok
        free_vram()
        log_vram("stage2 unloaded")
