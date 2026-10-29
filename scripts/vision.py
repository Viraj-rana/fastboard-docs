#!/usr/bin/env python3
"""Работа с локальной моделью Ollama: чат и распознавание изображений.

Общий модуль для бота (ответы, разбор присланных картинок) и индексатора
(вытаскивание текста из скриншотов документации).

Используется нативный API Ollama, а не OpenAI-совместимый: он принимает
картинки и умеет отключать режим рассуждений (`think: false`), без чего
qwen3.6 отдаёт пустой ответ.
"""

import base64
import json
import logging
import os
import re
import time

import requests

log = logging.getLogger("fb-rag")

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
VISION_MODEL = os.environ.get("VISION_MODEL", os.environ.get("OPENROUTER_CHAT_MODEL", "qwen3.6:27b-256k"))
CHAT_MODEL = os.environ.get("OPENROUTER_CHAT_MODEL", VISION_MODEL)
# Пусто = не переопределять контекст: модель на GPU уже загружена с большим
# контекстом, а смена num_ctx заставила бы Ollama перезагрузить модель.
NUM_CTX = os.environ.get("NUM_CTX", "").strip()
NUM_PREDICT = int(os.environ.get("NUM_PREDICT", "1400"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.2"))
# Один зависший запрос блокировал очередь на десять минут: при деградации
# скорости генерации таймаут в 600 секунд гарантированно вырабатывался целиком.
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "240"))
# Images stay on Ollama. This vLLM path is text-only.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama").strip().lower() or "ollama"
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "").strip()

DESCRIBE_PROMPT = (
    "Внимательно изучи изображение. Ответь на русском языке строго в таком формате:\n\n"
    "ОПИСАНИЕ: что изображено (тип интерфейса, экран, график, таблица, схема, фото — что именно видно).\n"
    "ТЕКСТ: полностью и дословно выпиши весь текст, который виден на изображении "
    "(надписи, заголовки, подписи осей, значения, пункты меню, сообщения об ошибках, SQL-запросы, числа). "
    "Сохраняй исходную формулировку и язык. Если текста нет — напиши «текста нет».\n"
    "СУТЬ: главное, что нужно понять из изображения (например, какая ошибка возникла "
    "или какие данные показаны)."
)

# Для индексации: нужен только полезный текст, без рассуждений о стиле картинки.
OCR_PROMPT = (
    "Это скриншот из документации BI-платформы Fastboard. Выпиши на русском языке:\n"
    "1) весь видимый текст дословно — названия полей, кнопок, пунктов меню, вкладок, "
    "подписи, значения, формулы, SQL;\n"
    "2) одной строкой — что показывает скриншот.\n"
    "Только текст с картинки и краткое пояснение, без вступлений и без домыслов. "
    "Если текста на изображении нет — ответь «нет текста»."
)


# Слот генерации на GPU один и делится с другими сервисами, поэтому запрос
# может не уложиться в таймаут просто из-за чужой очереди. Один повтор дешевле,
# чем потерянный вопрос пользователя.
RETRIES = int(os.environ.get("LLM_RETRIES", "1"))


def _with_retry(call, what):
    last = None
    for attempt in range(RETRIES + 1):
        try:
            return call()
        except (requests.Timeout, requests.ConnectionError) as e:
            last = e
            if attempt < RETRIES:
                print(f"  повтор запроса к модели ({what}): {type(e).__name__}")
                time.sleep(2)
    raise last


def _ollama_chat_native(messages, images=None, model=None, num_predict=None, temperature=None):
    msgs = [dict(m) for m in messages]
    if images:
        msgs[-1]["images"] = images
    options = {
        "temperature": TEMPERATURE if temperature is None else temperature,
        "num_predict": NUM_PREDICT if num_predict is None else num_predict,
    }
    if NUM_CTX:
        options["num_ctx"] = int(NUM_CTX)
    def call():
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": model or CHAT_MODEL, "messages": msgs,
                  "stream": False, "think": False, "keep_alive": -1,
                  "options": options},
            timeout=(15, LLM_TIMEOUT),
        )
        resp.raise_for_status()
        return resp

    body = _with_retry(call, "ответ").json()
    _log_ollama(body, None)
    content = (body.get("message") or {}).get("content", "")
    return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()


def _ollama_chat_stream_native(messages, model=None, num_predict=None, temperature=None, on_delta=None):
    options = {
        "temperature": TEMPERATURE if temperature is None else temperature,
        "num_predict": NUM_PREDICT if num_predict is None else num_predict,
    }
    if NUM_CTX:
        options["num_ctx"] = int(NUM_CTX)

    parts = []
    started = time.perf_counter()
    first = None
    done = {}

    def open_stream():
        return requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": model or CHAT_MODEL, "messages": [dict(m) for m in messages],
                  "stream": True, "think": False, "keep_alive": -1,
                  "options": options},
            timeout=(15, LLM_TIMEOUT), stream=True)

    # Повтор имеет смысл только пока не пришло ни одного куска ответа:
    # иначе пользователь увидел бы начало ответа дважды.
    with _with_retry(open_stream, "поток") as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            piece = (chunk.get("message") or {}).get("content", "")
            if piece:
                if first is None:
                    first = time.perf_counter() - started
                parts.append(piece)
                if on_delta:
                    try:
                        keep = on_delta("".join(parts))
                    except Exception:
                        keep = True
                    if keep is False:
                        break
            if chunk.get("done"):
                done = chunk
                break
    _log_ollama(done, first)
    return re.sub(r"<think>.*?</think>", "", "".join(parts), flags=re.S).strip()


def _log_ollama(body, first_s):
    if not body:
        return
    log.info(
        "ollama first=%.2fs prompt_eval=%.2fs eval=%.2fs",
        -1 if first_s is None else first_s,
        (body.get("prompt_eval_duration") or 0) / 1e9,
        (body.get("eval_duration") or 0) / 1e9,
    )


def _strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def _vllm_model(model):
    return VLLM_MODEL or model or CHAT_MODEL


def _vllm_body(messages, model, num_predict, temperature, stream):
    return {
        "model": _vllm_model(model),
        "messages": [dict(m) for m in messages],
        "stream": stream,
        "max_tokens": NUM_PREDICT if num_predict is None else num_predict,
        "temperature": TEMPERATURE if temperature is None else temperature,
        # qwen3.6 returns empty content unless thinking is off
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _vllm_content(body):
    choices = body.get("choices") or [{}]
    message = (choices[0] or {}).get("message") or {}
    return message.get("content") or ""


def _vllm_delta(chunk):
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = (choices[0] or {}).get("delta") or {}
    piece = delta.get("content")
    return piece if isinstance(piece, str) else ""


def _vllm_chat(messages, model=None, num_predict=None, temperature=None):
    def call():
        resp = requests.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json=_vllm_body(messages, model, num_predict, temperature, False),
            timeout=(15, LLM_TIMEOUT),
        )
        resp.raise_for_status()
        return resp

    return _strip_think(_vllm_content(_with_retry(call, "ответ").json()))


def _vllm_chat_stream(messages, model=None, num_predict=None, temperature=None, on_delta=None):
    parts = []

    def open_stream():
        return requests.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json=_vllm_body(messages, model, num_predict, temperature, True),
            timeout=(15, LLM_TIMEOUT),
            stream=True,
        )

    with _with_retry(open_stream, "поток") as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line:
                continue
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            piece = _vllm_delta(chunk)
            if not piece:
                continue
            parts.append(piece)
            if on_delta:
                try:
                    keep = on_delta("".join(parts))
                except Exception:
                    keep = True
                if keep is False:
                    resp.close()
                    break
    return _strip_think("".join(parts))


def ollama_chat(messages, images=None, model=None, num_predict=None, temperature=None):
    if images or LLM_BACKEND != "vllm":
        return _ollama_chat_native(messages, images, model, num_predict, temperature)
    return _vllm_chat(messages, model, num_predict, temperature)


def ollama_chat_stream(messages, model=None, num_predict=None, temperature=None, on_delta=None):
    if LLM_BACKEND != "vllm":
        return _ollama_chat_stream_native(messages, model, num_predict, temperature, on_delta)
    return _vllm_chat_stream(messages, model, num_predict, temperature, on_delta)


def describe_image(image_bytes, hint=""):
    """Разбор картинки, присланной пользователем: что на ней и какой текст."""
    prompt = DESCRIBE_PROMPT + (f"\n\nПользователь уточняет: {hint}" if hint else "")
    return _ollama_chat_native([{"role": "user", "content": prompt}],
                               images=[base64.b64encode(image_bytes).decode()], model=VISION_MODEL)


def ocr_image(image_bytes, num_predict=700):
    """Текст со скриншота документации — для индексации в базу знаний."""
    return _ollama_chat_native([{"role": "user", "content": OCR_PROMPT}],
                               images=[base64.b64encode(image_bytes).decode()],
                               model=VISION_MODEL, num_predict=num_predict)
