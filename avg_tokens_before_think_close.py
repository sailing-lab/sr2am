#!/usr/bin/env python3
import argparse
import json
import re
import sys
from collections import defaultdict
from typing import Any, Iterable


THINK_CLOSE = "</think>"

# Matches <tagname>...</tagname> blocks (non-greedy, dotall).
# Used by the "configurator" source to extract all tagged reasoning blocks.
_TAG_BLOCK_RE = re.compile(r"<(\w+)>.*?</\1>", re.DOTALL)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
                else:
                    c = item.get("content")
                    if isinstance(c, str):
                        parts.append(c)
        return "".join(parts)
    if isinstance(content, dict):
        t = content.get("text")
        if isinstance(t, str):
            return t
        c = content.get("content")
        if isinstance(c, str):
            return c
    return str(content)


def _extract_reasoning_texts(obj: Any, *, in_reasoning: bool = False) -> list[str]:
    texts: list[str] = []

    if isinstance(obj, list):
        for item in obj:
            texts.extend(_extract_reasoning_texts(item, in_reasoning=in_reasoning))
        return texts

    if isinstance(obj, dict):
        obj_type = obj.get("type")
        next_in_reasoning = in_reasoning or (obj_type == "reasoning")

        if obj_type == "reasoning_text":
            t = obj.get("text")
            if isinstance(t, str) and t:
                texts.append(t)

        # Some schemas may include the reasoning text directly on a reasoning item.
        if next_in_reasoning:
            rt = obj.get("reasoning_text")
            if isinstance(rt, str) and rt:
                texts.append(rt)

        texts.extend(_extract_reasoning_texts(obj.get("content"), in_reasoning=next_in_reasoning))
        return texts

    return texts


def _extract_configurator_reasoning(content: Any) -> str:
    """Return all <tag>...</tag> blocks from *content* concatenated together.

    In "configurator"-style assistant messages the model interleaves tagged
    reasoning blocks (e.g. <configurator>, <user_intent>, <summary>,
    <reflection>) with plain answer text.  Only the tagged blocks (including
    the opening/closing tags themselves) are counted as reasoning tokens.
    """
    text = _content_to_text(content)
    blocks = [m.group(0) for m in _TAG_BLOCK_RE.finditer(text)]
    return "".join(blocks)


def iter_jsonl(path: str, encoding: str = "utf-8") -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding=encoding, errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse JSON on line {line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise RuntimeError(f"Expected JSON object on line {line_no}, got {type(obj).__name__}")
            yield obj


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "For each row of a JSONL file, sum (over assistant messages) the number of tokens in "
            "assistant.content (or assistant.reasoning_content) BEFORE the first '</think>' tag (excluding the tag itself). "
            "Then print the average of that sum across rows."
        )
    )
    ap.add_argument(
        "--input",
        default=None,
        help="Path to JSONL input file.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Local model directory used to load the tokenizer.",
    )
    ap.add_argument("--encoding", default="utf-8", help="Input file encoding (default: utf-8).")
    ap.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="If >0, stop after processing this many rows (for quick tests).",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print a progress message to stderr every N rows (0 to disable).",
    )
    ap.add_argument(
        "--missing-behavior",
        choices=["zero", "all"],
        default="zero",
        help=(
            "If an assistant message lacks '</think>', count 0 tokens ('zero') or count all tokens in the message ('all')."
        ),
    )
    ap.add_argument(
        "--think-close-mode",
        choices=["content_only", "always", "never"],
        default="content_only",
        help=(
            "Whether to truncate at '</think>'. "
            "'content_only' truncates only when the selected text source is assistant.content (recommended); "
            "'always' truncates for both content and reasoning_content; "
            "'never' never truncates (counts full selected text)."
        ),
    )
    ap.add_argument(
        "--assistant-text-source",
        choices=["content", "reasoning_content", "prefer_reasoning_content", "response_api", "response_api_usage", "configurator"],
        default="prefer_reasoning_content",
        help=(
            "Which assistant field to tokenize. "
            "'content' uses message['content']; "
            "'reasoning_content' uses message['reasoning_content']; "
            "'prefer_reasoning_content' uses reasoning_content when present, else falls back to content; "
            "'response_api' extracts reasoning_text from response-api style messages where item type is 'reasoning'; "
            "'response_api_usage' reads reasoning token counts directly from message['usage']['output_tokens_details']['reasoning_tokens'] for messages that contain a 'usage' field; "
            "'configurator' extracts all <tag>...</tag> blocks from message['content'] (the tags themselves are included)."
        ),
    )
    ap.add_argument(
        "--breakdown-by-dataset",
        action="store_true",
        help="If set, also print per-dataset averages using row['dataset'].",
    )
    ap.add_argument(
        "--dataset-field",
        default="dataset",
        help="Top-level field name to use for dataset grouping (default: dataset).",
    )
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:  # pragma: no cover
        print(
            "ERROR: failed to import transformers. Install it (e.g. pip install transformers).",
            file=sys.stderr,
        )
        raise

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True,
        )
    except Exception as e:
        print(f"ERROR: failed to load tokenizer from {args.model!r}: {e}", file=sys.stderr)
        return 2

    total_rows = 0
    total_tokens = 0
    total_reasoning_tokens = 0
    assistant_msgs = 0
    assistant_msgs_with_close = 0
    assistant_msgs_missing_close = 0
    assistant_msgs_used_reasoning_content = 0

    ds_rows: dict[str, int] = defaultdict(int)
    ds_tokens: dict[str, int] = defaultdict(int)
    ds_reasoning_tokens: dict[str, int] = defaultdict(int)

    for row in iter_jsonl(args.input, encoding=args.encoding):
        total_rows += 1
        row_tokens = 0
        row_reasoning_tokens = 0

        msgs = row.get("messages", [])
        if isinstance(msgs, list):
            for m in msgs:
                if not isinstance(m, dict):
                    continue

                item_type = m.get("type")
                # gpt-oss response-api rows place reasoning in standalone items:
                # {"type": "reasoning", "content": [{"type": "reasoning_text", "text": ...}, ...]}
                if item_type == "reasoning":
                    reasoning_parts = _extract_reasoning_texts(m)
                    if reasoning_parts:
                        reasoning_text = "".join(reasoning_parts)
                        # print('reasoning_text:', reasoning_text)
                        row_reasoning_tokens += len(tokenizer.encode(reasoning_text, add_special_tokens=False))

                if args.assistant_text_source == "response_api_usage":
                    usage = m.get("usage")
                    if isinstance(usage, dict):
                        reasoning_tokens = (
                            usage.get("output_tokens_details", {}) or {}
                        ).get("reasoning_tokens")
                        if isinstance(reasoning_tokens, (int, float)):
                            row_reasoning_tokens += int(reasoning_tokens)
                    continue

                if args.assistant_text_source == "response_api":
                    continue

                if m.get("role") != "assistant":
                    continue
                assistant_msgs += 1

                used_reasoning = False
                if args.assistant_text_source == "content":
                    text = _content_to_text(m.get("content"))
                elif args.assistant_text_source == "reasoning_content":
                    text = _content_to_text(m.get("reasoning_content"))
                    used_reasoning = True
                elif args.assistant_text_source == "configurator":
                    text = _extract_configurator_reasoning(m.get("content"))
                    used_reasoning = True
                else:  # prefer_reasoning_content
                    rc = m.get("reasoning_content")
                    if rc is not None:
                        text = _content_to_text(rc)
                        used_reasoning = True
                    else:
                        text = _content_to_text(m.get("content"))

                if used_reasoning:
                    assistant_msgs_used_reasoning_content += 1

                if not text:
                    continue

                if args.think_close_mode == "always":
                    truncate = True
                elif args.think_close_mode == "never":
                    truncate = False
                else:  # content_only
                    truncate = not used_reasoning

                if truncate:
                    idx = text.find(THINK_CLOSE)
                    if idx >= 0:
                        assistant_msgs_with_close += 1
                        prefix = text[:idx]
                    else:
                        assistant_msgs_missing_close += 1
                        if args.missing_behavior == "all":
                            prefix = text
                        else:
                            prefix = ""
                else:
                    prefix = text

                if prefix:
                    row_tokens += len(tokenizer.encode(prefix, add_special_tokens=False))

        if args.assistant_text_source in ("response_api", "response_api_usage"):
            row_tokens = row_reasoning_tokens

        total_tokens += row_tokens
        total_reasoning_tokens += row_reasoning_tokens
        if args.breakdown_by_dataset:
            ds = row.get(args.dataset_field, "<missing>")
            if not isinstance(ds, str) or not ds:
                ds = str(ds) if ds is not None else "<missing>"
            ds_rows[ds] += 1
            ds_tokens[ds] += row_tokens
            ds_reasoning_tokens[ds] += row_reasoning_tokens

        if args.progress_every and (total_rows % args.progress_every == 0):
            print(
                f"processed_rows={total_rows} total_tokens_before_think_close={total_tokens} "
                f"total_reasoning_tokens={total_reasoning_tokens} assistant_msgs={assistant_msgs} "
                f"with_close={assistant_msgs_with_close} missing_close={assistant_msgs_missing_close}",
                file=sys.stderr,
                flush=True,
            )

        if args.max_rows and total_rows >= args.max_rows:
            break

    if total_rows == 0:
        print("No rows processed (empty JSONL?).", file=sys.stderr)
        return 1

    avg = total_tokens / total_rows
    avg_reasoning = total_reasoning_tokens / total_rows
    print(f"rows={total_rows}")
    print(f"total_tokens_before_think_close={total_tokens}")
    print(f"avg_tokens_before_think_close_per_row={avg:.6f}")
    print(f"total_reasoning_tokens={total_reasoning_tokens}")
    print(f"avg_reasoning_tokens_per_row={avg_reasoning:.6f}")
    print(f"assistant_msgs={assistant_msgs}")
    print(f"assistant_msgs_with_think_close={assistant_msgs_with_close}")
    print(f"assistant_msgs_missing_think_close={assistant_msgs_missing_close}")
    print(f"assistant_msgs_used_reasoning_content={assistant_msgs_used_reasoning_content}")
    print(f"assistant_text_source={args.assistant_text_source}")
    print(f"think_close_mode={args.think_close_mode}")
    print(f"missing_behavior={args.missing_behavior}")

    if args.breakdown_by_dataset:
        print("per_dataset:")
        for ds, n in sorted(ds_rows.items(), key=lambda kv: (-kv[1], kv[0])):
            tok = ds_tokens.get(ds, 0)
            reasoning_tok = ds_reasoning_tokens.get(ds, 0)
            avg_ds = (tok / n) if n else 0.0
            avg_reasoning_ds = (reasoning_tok / n) if n else 0.0
            print(
                f"  dataset={ds}\trows={n}\ttotal_tokens={tok}\tavg_tokens_per_row={avg_ds:.6f}"
                f"\ttotal_reasoning_tokens={reasoning_tok}\tavg_reasoning_tokens_per_row={avg_reasoning_ds:.6f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

