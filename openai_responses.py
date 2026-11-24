"""OpenAI Responses API에 스키마가 고정된 JSON 응답을 요청한다."""

import json
import time
import urllib.error
import urllib.request


RESPONSES_URL = "https://api.openai.com/v1/responses"
RETRYABLE_HTTP = {408, 409, 429, 500, 502, 503, 504}


def strict_object(properties):
    # Structured Outputs에서 객체 필드는 모두 required로 선언한다.
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def uses_reasoning(model):
    return model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))


def request_settings(model, reasoning_effort, temperature, max_output_tokens):
    settings = {"max_output_tokens": max_output_tokens}
    if uses_reasoning(model):
        settings["reasoning"] = {"effort": reasoning_effort}
    else:
        settings["temperature"] = temperature
    return settings


def error_fields(error):
    try:
        fields = json.loads(error.read()).get("error", {})
    except (OSError, ValueError, AttributeError):
        return {}
    return fields if isinstance(fields, dict) else {}


def call_openai_json(api_key, model, system_prompt, user_prompt, *, schema_name, schema, settings, timeout=120):
    payload = {
        "model": model, "store": False,
        "input": [{"role": "developer", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "text": {"format": {"type": "json_schema", "name": schema_name, "strict": True, "schema": schema}},
        **settings,
    }
    request = urllib.request.Request(
        RESPONSES_URL, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}, method="POST",
    )
    for retry in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.load(response)
                request_id = response.headers.get("x-request-id")
            break
        except urllib.error.HTTPError as exc:
            fields = error_fields(exc)
            code, param = fields.get("code"), fields.get("param")
            headers = exc.headers or {}
            if retry == 2 or exc.code not in RETRYABLE_HTTP or code == "insufficient_quota":
                detail = ", ".join(str(value) for value in (code, param, headers.get("x-request-id")) if value)
                raise RuntimeError(f"OpenAI HTTP {exc.code} ({model})" + (f" | {detail}" if detail else "")) from exc
            retry_after = headers.get("Retry-After", "")
            delay = min(60, int(retry_after)) if retry_after.isdecimal() else 2 ** retry
        except (urllib.error.URLError, TimeoutError) as exc:
            if retry == 2:
                raise RuntimeError(f"OpenAI 연결 실패 ({model})") from exc
            delay = 2 ** retry
        except (UnicodeError, ValueError) as exc:
            raise RuntimeError(f"OpenAI 응답 본문을 읽을 수 없습니다 ({model})") from exc
        time.sleep(delay)

    if not isinstance(body, dict):
        raise RuntimeError(f"OpenAI 응답 객체의 형식이 올바르지 않습니다 ({model})")
    response_usage = body.get("usage") or {}
    usage = {
        "prompt_tokens": response_usage.get("input_tokens", 0),
        "completion_tokens": response_usage.get("output_tokens", 0),
        "reasoning_tokens": (response_usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0),
        "cached_prompt_tokens": (response_usage.get("input_tokens_details") or {}).get("cached_tokens", 0),
        "model": body.get("model", model), "response_id": body.get("id"), "request_id": request_id,
    }
    status = body.get("status")
    if status != "completed":
        usage["api_error"] = "api_incomplete" if status == "incomplete" else "api_response_failed"
        usage["api_error_detail"] = (body.get("incomplete_details") or {}).get("reason") or (body.get("error") or {}).get("code")
        return None, usage

    # 추론 등 다른 출력 항목을 건너뛰고 실제 메시지의 텍스트와 거절을 찾는다.
    try:
        parts = [part for item in body.get("output", []) if item.get("type") == "message"
                 for part in item.get("content", [])]
        if any(part.get("type") == "refusal" for part in parts):
            return None, {**usage, "api_error": "api_refusal"}
        value = json.loads("".join(part["text"] for part in parts if part.get("type") == "output_text"))
    except (ValueError, KeyError, TypeError, AttributeError):
        return None, {**usage, "api_error": "api_invalid_output"}
    if not isinstance(value, dict):
        return None, {**usage, "api_error": "api_invalid_output"}
    return value, usage
