import json

import evaluate
import pytest


def calculate_rouge_score(predictions, references):
    """
    Compute ROUGE score between predictions and references.
    Args:
        predictions: List of predictions.
        references: List of references.
    Returns:
        ROUGE score. Format: {
            "rouge1": float,
            "rouge2": float,
            "rougeL": float,
            "rougeLsum": float
        }
    """
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=predictions, references=references)


def check_rouge_score(output_json_str, reference_json_str):
    """
    Calculate ROUGE score between output and reference.
    Args:
        output_json_str: String of the output JSON.
        reference_json_str: String of the reference JSON.
    """
    try:
        output_json = json.loads(output_json_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse output JSON: {e}. Content preview: '{output_json_str[:200]}...'"
        )

    try:
        reference_json = json.loads(reference_json_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse reference JSON: {e}. Content preview: '{reference_json_str[:200]}...'"
        )

    references = []
    predictions = []
    for message in reference_json["messages"]:
        references.append(message["reference"])
    for response in output_json["responses"]:
        predictions.append(response["output_text"])

    rouge_score = calculate_rouge_score(predictions, references)
    if rouge_score["rouge1"] < 0.25 or rouge_score["rougeL"] < 0.20:
        pytest.fail(
            f"Rouge1 (threshold: 0.25) or RougeL (threshold: 0.20) score is too low. Rouge1 score: {rouge_score['rouge1']}. RougeL score: {rouge_score['rougeL']}. references: {references}. predictions: {predictions}"
        )
    return rouge_score
