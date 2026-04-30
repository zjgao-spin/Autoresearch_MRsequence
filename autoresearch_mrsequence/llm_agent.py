"""LLM Agent client via OpenRouter API — reads AGENTS.md and proposes parameters."""

import json, re, time, copy
import requests

OPENROUTER_PRICING = {
    # USD per million tokens (input, output, reasoning)
    "deepseek/deepseek-v4-pro": (2.25, 9.00, 9.00),
    "moonshotai/kimi-k2.6": (2.00, 8.00, 8.00),
    "z-ai/glm-5": (0.55, 2.19, 2.19),
    "qwen/qwen3.6-max-preview": (1.20, 4.80, 4.80),
    "xiaomi/mimo-v2.5-pro": (0.40, 1.50, 1.50),
}


class LLMAgent:
    """OpenRouter LLM client for MRI sequence parameter proposal."""

    def __init__(self, model_id, api_key, program_md, editable_meta, fixed_params):
        self.model = model_id
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.program_md = program_md
        self.editable_meta = editable_meta
        self.fixed_params = fixed_params
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_reasoning = 0
        self.total_cost = 0.0
        self.calls = 0

    def get_cost_info(self):
        """Return dict with token and cost summary."""
        return {
            "calls": self.calls,
            "tokens_in": self.total_tokens_in,
            "tokens_out": self.total_tokens_out,
            "reasoning": self.total_reasoning,
            "cost": round(self.total_cost, 6),
        }

    def propose(self, best_params, history, exp_num):
        """Send prompt to LLM, parse JSON response, return (params, stats)."""
        prompt = self._build_prompt(best_params, history, exp_num)

        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/zjgao-spin/Autoresearch_MRsequence",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.6,
                    "max_tokens": 1024,
                },
                timeout=120,
            )

            if r.status_code != 200:
                err = r.json().get("error", {}).get("message", r.text[:200])
                raise RuntimeError(f"API error {r.status_code}: {err}")

            data = r.json()
        except requests.exceptions.Timeout:
            # Fallback: random proposal
            return self._fallback_random(best_params), {"tokens_in": 0, "tokens_out": 0, "reasoning": 0, "error": "timeout"}
        except Exception as e:
            return self._fallback_random(best_params), {"tokens_in": 0, "tokens_out": 0, "reasoning": 0, "error": str(e)[:100]}

        usage = data.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)
        reasoning = usage.get("reasoning_tokens", 0) or usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)

        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        self.total_reasoning += reasoning
        self.calls += 1

        # Compute cost (USD per million tokens)
        prices = OPENROUTER_PRICING.get(self.model, (0.0, 0.0, 0.0))
        self.total_cost += (tokens_in / 1e6) * prices[0] + (tokens_out / 1e6) * prices[1] + (reasoning / 1e6) * prices[2]

        content = data["choices"][0]["message"].get("content")
        if not content or not content.strip():
            # Some models return empty content with reasoning tokens only
            # or the model refused to answer. Fall back to random.
            content = data["choices"][0]["message"].get("reasoning", "")
            if not content:
                return self._fallback_random(best_params), {
                    "tokens_in": tokens_in, "tokens_out": tokens_out,
                    "reasoning": reasoning, "error": "empty_response"
                }

        # Parse JSON from response
        try:
            proposed = self._parse_json(content)
        except Exception:
            return self._fallback_random(best_params), {
                "tokens_in": tokens_in, "tokens_out": tokens_out,
                "reasoning": reasoning, "error": "json_parse_failed"
            }

        # Merge with best_params, validate against editable_meta
        params = copy.deepcopy(best_params)
        for k, v in proposed.items():
            if k in self.editable_meta:
                params[k] = self._clip(k, v)

        stats = {"tokens_in": tokens_in, "tokens_out": tokens_out, "reasoning": reasoning}
        return params, stats

    def _build_prompt(self, best_params, history, exp_num):
        meta_desc = {}
        for k, v in self.editable_meta.items():
            t = v["type"]
            if t == "int":
                rng = v.get("valid", v.get("range", "?"))
                meta_desc[k] = f"int: {rng}"
            elif t == "float":
                meta_desc[k] = f"float: [{v['range'][0]}, {v['range'][1]}]"
            elif t == "choice":
                meta_desc[k] = f"choice: {v['choices']}"
            elif t == "list":
                rng = v["range"]
                meta_desc[k] = f"list[{rng[0]}-{rng[1]}] (length = n_echo)"

        hist_lines = []
        for h in history[-8:]:
            p = h.get("params", {})
            flips = p.get("rf_flip_angles", [])
            flips_str = f"[{','.join(str(int(f)) for f in flips[:4])}...]" if len(flips) > 4 else str(flips)
            hist_lines.append(
                f"  Exp#{h['exp']}: {h.get('status','?')}  "
                f"MAE={h.get('mae',0):.4f}  Score={h.get('score',0):.4f}  "
                f"turbo={p.get('n_echo','?')} enc={p.get('encoding','?')} flips={flips_str}"
            )

        return f"""You are an MRI sequence optimization agent running on the MRzero Bloch simulator.

## AGENT PROTOCOL (domain knowledge)
{self.program_md}

## TASK
Optimize a 2D Turbo Spin Echo (TSE) sequence to minimize the composite score:
  Score = 0.5*MAE/baseline + 0.3*SAR/baseline + 0.2*Time/baseline

## FIXED PARAMETERS (from instruction, DO NOT CHANGE)
{json.dumps(self.fixed_params, indent=2)}

## EDITABLE PARAMETERS
{json.dumps(meta_desc, indent=2)}

## CURRENT BEST
{json.dumps(best_params, indent=2, default=str)}

## RECENT EXPERIMENT HISTORY
{chr(10).join(hist_lines) if hist_lines else "(none yet)"}

## INSTRUCTIONS
1. Analyze the history to understand what worked and what didn't.
2. Propose NEW parameters for experiment #{exp_num}.
3. Vary 1-3 parameters at a time.
4. Return ONLY a valid JSON object with the parameters you want to change."""

    def _parse_json(self, text):
        text = text.strip()
        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try extract from markdown code block
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        # Try find first { ... } block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"Cannot parse JSON from: {text[:200]}")

    def _clip(self, key, value):
        meta = self.editable_meta.get(key, {})
        t = meta.get("type", "")
        if t == "int":
            if "valid" in meta and isinstance(value, int):
                if value in meta["valid"]:
                    return value
                return min(meta["valid"], key=lambda x: abs(x - value))
            lo, hi = meta.get("range", [0, 999])
            return int(max(lo, min(hi, value)))
        elif t == "float":
            lo, hi = meta.get("range", [0, 1])
            return round(max(lo, min(hi, float(value))), 4)
        elif t == "choice":
            if value in meta.get("choices", []):
                return value
            return meta.get("choices", [value])[0]
        elif t == "list":
            lo, hi = meta.get("range", [20, 180])
            mag = meta.get("perturb_mag", 15)
            result = []
            for v in (value if isinstance(value, list) else [value]):
                result.append(int(max(lo, min(hi, round(float(v))))))
            return result
        return value

    def _fallback_random(self, best_params):
        """Generate a random proposal as fallback when LLM fails."""
        import random
        p = copy.deepcopy(best_params)
        keys = list(self.editable_meta.keys())
        if not keys:
            return p
        for key in random.sample(keys, min(random.randint(1, 3), len(keys))):
            meta = self.editable_meta[key]
            t = meta["type"]
            if t == "int":
                p[key] = random.choice(meta["valid"]) if "valid" in meta else random.randint(*meta["range"])
            elif t == "float":
                lo, hi = meta["range"]
                p[key] = round(random.uniform(lo, hi), 4)
            elif t == "choice":
                p[key] = random.choice(meta["choices"])
            elif t == "list":
                lo, hi = meta["range"]
                n = len(p[key])
                p[key] = [round(random.uniform(lo, hi)) for _ in range(n)]
        return p
