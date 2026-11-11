"""Connect the project's real Qwen inference code here.

The uploaded repository currently contains the BM25/Dense/Hybrid retrieval
pipeline but no Qwen loading or answer-generation module. Replace this
NotImplementedError by importing and calling that real function.

Signature:
    generate_answer(question: str, contexts: list[dict]) -> dict

Return:
    {
      "answer": "generated Turkish answer",
      "citations": [
        {"document_id": "source document id", "article": "article number",
         "paragraph": "paragraph number (optional)"}
      ]
    }
"""
def generate_answer(question: str, contexts: list[dict]) -> dict:
    raise NotImplementedError(
        "Qwen generation function is not present in the uploaded repository. "
        "Connect your actual model inference code here."
    )
