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

    # Check thresholds and provide detailed failure information
    rouge1_threshold = 0.25
    rougeL_threshold = 0.20

    if rouge_score["rouge1"] < rouge1_threshold or rouge_score[
            "rougeL"] < rougeL_threshold:
        failure_details = [
            f"ROUGE score below threshold",
            f"Rouge1: {rouge_score['rouge1']:.4f} (threshold: {rouge1_threshold})",
            f"RougeL: {rouge_score['rougeL']:.4f} (threshold: {rougeL_threshold})",
            f"Number of predictions: {len(predictions)}",
            f"Number of references: {len(references)}"
        ]

        # Add sample predictions/references for debugging (limit output)
        if predictions:
            failure_details.append(
                f"Sample prediction: {predictions[0][:100]}...")
        if references:
            failure_details.append(
                f"Sample reference: {references[0][:100]}...")

        pytest.fail("\n".join(failure_details))

    return rouge_score
