"""Quick evaluation: answer correctness (LLM judge), refusal accuracy and citation coverage.

Usage: python -m eval.run_eval   -> writes eval/results.md
"""
import json
from pathlib import Path

from app import config, llm, rag

JUDGE_SYSTEM = "You grade a support bot. Reply only YES or NO."

JUDGE_PROMPT = """Question: {question}
Expected answer: {expected}
Bot answer: {answer}

Does the bot answer contain the expected answer (same meaning, extra detail is fine)? Reply only YES or NO."""


def is_correct(q: dict, result: dict) -> bool:
    if not q["in_docs"]:
        return not result["found"]
    if not result["found"]:
        return False
    verdict = llm.complete(JUDGE_SYSTEM, JUDGE_PROMPT.format(answer=result["answer"], **q), max_tokens=1024)
    return verdict.strip().upper().startswith("YES")


def main() -> None:
    questions = json.loads(Path("eval/questions.json").read_text())
    rows, correct, cited, answerable = [], 0, 0, 0
    for q in questions:
        result = rag.answer(q["question"])
        ok = is_correct(q, result)
        correct += ok
        if q["in_docs"]:
            answerable += 1
            cited += bool(result["sources"])
        sources = ", ".join(f"{s['document']} p.{s['page']}" for s in result["sources"]) or "-"
        answer = result["answer"].replace("\n", " ").replace("|", "\\|")
        rows.append(f"| {'✅' if ok else '❌'} | {q['question']} | {answer} | {sources} |")
        print(("PASS " if ok else "FAIL ") + q["question"])

    summary = [
        "# Evaluation results",
        "",
        f"- Model: `{config.CHAT_MODEL}`, embeddings: `{config.EMBEDDING_MODEL}`, "
        f"top_k={config.TOP_K}, min_score={config.MIN_SCORE}",
        f"- **Accuracy: {correct}/{len(questions)}** (answerable judged by LLM; "
        "out-of-scope must be refused)",
        f"- **Citation coverage: {cited}/{answerable}** answerable questions returned sources",
        "",
        "| | Question | Answer | Sources |",
        "|---|---|---|---|",
        *rows,
    ]
    Path("eval/results.md").write_text("\n".join(summary) + "\n")
    print(f"\nAccuracy {correct}/{len(questions)} — wrote eval/results.md")


if __name__ == "__main__":
    main()
