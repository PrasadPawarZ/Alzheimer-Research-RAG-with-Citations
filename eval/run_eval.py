"""
Stretch goal: eval harness.

Scores:
  - retrieval@k: did the expected document appear anywhere in the top_k
    retrieved chunks?
  - refusal correctness: for the adversarial question (expected_doc_id=null),
    did the system correctly refuse instead of hallucinating?

Run from the project root:
    python eval/run_eval.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag_core

EVAL_PATH = os.path.join(os.path.dirname(__file__), "eval_set.json")


def main(top_k: int = 5):
    with open(EVAL_PATH) as f:
        eval_set = json.load(f)

    hits, total_retrieval_cases = 0, 0
    refusal_correct, refusal_cases = 0, 0

    print(f"{'ID':<4} {'Result':<10} {'Question'}")
    print("-" * 90)

    for item in eval_set:
        q = item["question"]
        expected = item["expected_doc_id"]

        citations = rag_core.retrieve(q, top_k=top_k)
        retrieved_doc_ids = {c.doc_id for c in citations}

        if expected is None:
            refusal_cases += 1
            result = rag_core.answer_question(q, top_k=top_k)
            ok = not result.covered
            refusal_correct += int(ok)
            print(f"{item['id']:<4} {'PASS' if ok else 'FAIL':<10} {q[:70]}")
        else:
            total_retrieval_cases += 1
            ok = expected in retrieved_doc_ids
            hits += int(ok)
            print(f"{item['id']:<4} {'HIT' if ok else 'MISS':<10} {q[:70]}")

    print("-" * 90)
    if total_retrieval_cases:
        print(f"Retrieval@{top_k}: {hits}/{total_retrieval_cases} = {hits/total_retrieval_cases:.1%}")
    if refusal_cases:
        print(f"Correct refusals: {refusal_correct}/{refusal_cases} = {refusal_correct/refusal_cases:.1%}")


if __name__ == "__main__":
    main()
