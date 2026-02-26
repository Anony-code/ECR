
from __future__ import annotations
import json
from typing import Dict, Any, List
from typing import Optional


def _json_dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2)


def _memory_compact(memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep the prompt memory payload stable and not too large.
    We include BOTH knowledge_core and experience_memory as you requested.
    """
    return {
        "knowledge_core": memory.get("knowledge_core", {}),
        "experience_knowledge": memory.get("experience_knowledge", {}),

    }


fullname = "Parkinson's disease"
shortname = "PD"


def clinical_reasoning_prompt(
    base_codes_diagnosis: List[str],
    base_codes_medication: List[str],
    delta_codes_diagnosis: List[str],
    delta_codes_medication: List[str],
    memory: Dict[str, Any],
    sex: Optional[str] = None,
    age: Optional[int] = None
) -> str:

    mem = _memory_compact(memory)
    year = 5

    if age is None:


        return (
            f"You are a national-level experienced physician with expertise in idiopathic {fullname} ({shortname}). "
            "You are interpreting EHR records to understand patient-level clinical patterns. "
            f"You are evaluating the patient at the PREDICTION TIME (baseline time), which is exactly {year} years before the future outcome window. "
            "You have access to the patient’s EHR history available up to this baseline time. "
            f"For training purposes only, you are also shown follow-up information (new observations that just appear from this time ({year} years prior to the future outcome window) to 1 year prior to the outcome window). "
            "This follow-up information is provided solely to help you understand, at a population level, how baseline patterns may develop and thus be interpreted well. "


            f"Your task is to interpret whether the baseline EHR pattern contains PD-relevant clinical anchors so that the patient would develop into {shortname} at the outcome window.\n\n"


            "You will review the patient’s EHR history at the baseline time: "
            "e.g., whether the history reflects motor, non-motor, cognitive, functional, behavioral, neuropsychiatric, or other systemic patterns, "
            f"whether such patterns appear patient-specific rather than generic, and whether these associate with {shortname} risk.\n\n"

            "Write ONE concise teaching-style clinical note as if you are explaining your reasoning to a medical student. "
            "Use clear, stepwise medical reasoning in full sentences, written as a single paragraph (do NOT use lists or headings). "
            "Explicitly name relevant diseases or clinical concepts, when appropriate, related to specific BASELINE evidence. "
            "Include (i) the most important baseline clinical patterns, "
            "(ii) plausible clinical interpretation and why it fits the baseline record.\n\n"

            "After completing the reasoning, you MUST end the paragraph with ONE final sentence labeled exactly as:\n\n"
            "\"Baseline clinical interpretation: ...\"\n\n"

            "Notes:\n"
            "- Do NOT center your reasoning on the absence of documented motor exams or rating scales. The main reasoning must focus on interpreting what IS present in the baseline record and how salient each pattern is.\n"
            "- No need to mention dependence on other data modalities except EHR, as EHR is the only information available.\n"
            "- In 'Baseline clinical interpretation', state what the record is primarily about in this patient (the important patterns).\n"
            "- Write this interpretation strictly from a retrospective descriptive perspective, not as if advising on future outcomes.\n"
            "- Unrecorded information may be treated as unknown rather than unexisting.\n\n"

            "Content guidelines for \"Baseline clinical interpretation\":\n"
            "- Describe overarching clinical patterns or meanings inferred from the baseline record.\n"
            "- Use explanatory language that interprets patterns rather than enumerating diagnoses or codes.\n"


            f"- Do NOT state whether the patient will or will not develop {shortname}.\n\n"

            f"Baseline diagnoses (available at prediction time): {base_codes_diagnosis or []}\n"
            f"Baseline medications (available at prediction time): {base_codes_medication or []}\n\n"

            f"Follow-up diagnoses (new observation from baseline to 1y prior to outcome window, training-only): {delta_codes_diagnosis or []}\n"
            f"Follow-up medications  (new observation from baseline to 1y prior to outcome window,, training-only): {delta_codes_medication or []}\n\n"

            "Existing clinician knowledge (background only; do not override patient-specific baseline evidence):\n"
            f"{mem}\n\n"

            "Write only the final clinical reasoning paragraph."
        )

    else:

        return (
            f"You are a national-level experienced physician with expertise in idiopathic {fullname} ({shortname}). "
            "You are interpreting EHR records to understand patient-level clinical patterns. "
            f"You are evaluating the patient at the PREDICTION TIME (baseline time), which is exactly {year} years before the future outcome window. "
            "You have access to the patient’s EHR history available up to this baseline time. "
            f"For training purposes only, you are also shown follow-up information (new observations that just appear from this time ({year} years prior to the future outcome window) to 1 year prior to the outcome window). "
            "This follow-up information is provided solely to help you understand, at a population level, how baseline patterns may develop and thus be interpreted well. "


            f"Your task is to interpret whether the baseline EHR pattern contains PD-relevant clinical anchors so that the patient would develop into {shortname} at the outcome window.\n\n"


            "You will review the patient’s EHR history at the baseline time: "
            "e.g., whether the history reflects motor, non-motor, cognitive, functional, behavioral, neuropsychiatric, or other systemic patterns, "
            f"whether such patterns appear patient-specific rather than generic, and whether these associate with {shortname} risk.\n\n"

            "Write ONE concise teaching-style clinical note as if you are explaining your reasoning to a medical student. "
            "Use clear, stepwise medical reasoning in full sentences, written as a single paragraph (do NOT use lists or headings). "
            "Explicitly name relevant diseases or clinical concepts, when appropriate, related to specific BASELINE evidence. "
            "Include (i) the most important baseline clinical patterns, "
            "(ii) plausible clinical interpretation and why it fits the baseline record.\n\n"

            "After completing the reasoning, you MUST end the paragraph with ONE final sentence labeled exactly as:\n\n"
            "\"Baseline clinical interpretation: ...\"\n\n"

            "Notes:\n"
            "- Do NOT center your reasoning on the absence of documented motor exams or rating scales. The main reasoning must focus on interpreting what IS present in the baseline record and how salient each pattern is.\n"
            "- No need to mention dependence on other data modalities except EHR, as EHR is the only information available.\n"
            "- In 'Baseline clinical interpretation', state what the record is primarily about in this patient (the important patterns).\n"
            "- Write this interpretation strictly from a retrospective descriptive perspective, not as if advising on future outcomes.\n\n"

            "Content guidelines for \"Baseline clinical interpretation\":\n"
            "- Describe overarching clinical patterns or meanings inferred from the baseline record.\n"
            "- Use explanatory language that interprets patterns rather than enumerating diagnoses or codes.\n"


            f"- Do NOT state whether the patient will or will not develop {shortname}.\n\n"
            f"Baseline demographics: age {age}, gender {sex}.\n"

            f"Baseline diagnoses (available at prediction time): {base_codes_diagnosis or []}\n"
            f"Baseline medications (available at prediction time): {base_codes_medication or []}\n\n"

            f"Follow-up diagnoses (new observation from baseline to 1y prior to outcome window, training-only): {delta_codes_diagnosis or []}\n"
            f"Follow-up medications  (new observation from baseline to 1y prior to outcome window,, training-only): {delta_codes_medication or []}\n\n"

            "Existing clinician knowledge (background only; do not override patient-specific baseline evidence):\n"
            f"{mem}\n\n"

            "Write only the final clinical reasoning paragraph."
        )


def calibration_update_prompt(
    memory: Dict[str, Any],
    reasoning_card: str,
    label: int,
) -> str:
    """
    Phase1 memory update prompt (fully unstructured text):
    """
    mem = _memory_compact(memory)
    year = 5


    return (
    f"You are a junior physician specializing in idiopathic {fullname} ({shortname}) and early clinical risk interpretation. "
    f"You previously evaluated a patient at the PREDICTION TIME (baseline time), which is exactly {year} years before the outcome window, "
    "using baseline EHR information only, and produced a clinical reasoning paragraph interpreting baseline patterns.\n\n"

    f"A senior physician has now revealed the true outcome label indicating whether the patient developed {shortname} at the outcome window. "
    "This supervision is provided to help you reflect on your baseline interpretation process and identify where your weighting or sensitivity "
    "to certain types of baseline patterns may have been miscalibrated.\n\n"

    "Your task is to write a reflection for this patient that analyzes how your prior baseline reasoning "
    "aligned or misaligned with the observed outcome. "
    "This reflection represents a single experience sample and will later be aggregated with reflections from other patients "
    "to update general clinician knowledge in terms of experience.\n\n"

    "Important guidance:\n"

    "- Focus on how you should have adjusted the salience or weighting of certain baseline clinical patterns, \n"
    "- Phrase your reflection in a way that could later contribute to cross-patient experience consolidation.\n\n"

    "Existing clinician knowledge (shared reference from previously seen patients; do not treat this as final):\n"
    f"{mem}\n\n"

    "Your prior reasoning paragraph for this patient:\n"
    f"{reasoning_card}\n\n"

    f"Observed outcome label (0 = no {shortname} at outcome window, 1 = {shortname} occured at outcome window): {int(label)}\n\n"

    "Constraints:\n"
    "- Write exactly ONE concise paragraph.\n"
    "- Do NOT use lists, numbering, headings, or special formatting.\n"


    "Write the reflection paragraph now."
)


def memory_refresh_prompt(
    new_calib_paragraphs: list[str],
    new_reason_paragraphs: list[str],
    memory: Dict[str, Any],
) -> str:
    """
    Refresh experience_memory.calibration_lessons using recent calibration paragraphs.
    Output: ONE short paragraph (no structure) that will become the single lesson stored.
    """
    reason_cards = [i.split('assistantfinal')[-1] for i in new_reason_paragraphs]
    calib_cards = [i.split('assistantfinal')[-1] for i in new_calib_paragraphs]


    joined_reason = "\n\n".join(reason_cards)
    joined = "\n\n".join(calib_cards)
    mem = _memory_compact(memory)


    return (
        "You are a physician specializing in neurodegenerative diseases who has evaluated many patients using EHR data "
        f"and has repeatedly calibrated your clinical judgments against true idiopathic {fullname} ({shortname}) outcomes. "
        "Over time, you have accumulated experience about which clinical patterns tend to be reliable and which tend to mislead. "
        "Your task now is to maintain and refine a clinician experience summary that represents how an experienced doctor "
        "internally adjusts intuition after seeing many outcomes.\n\n"

        "This clinician experience is NOT a case summary. "
        "It is a distilled set of clinical instincts, for example, that guide how evidence should be weighted, how confidence should be calibrated, "
        f"and how common reasoning traps should be avoided when judging whether a patient has {shortname} at the current visit.\n\n"

        "You should think like a senior doctor asking:\n"
        "E.g., across many patients, what patterns consistently deserve more trust? "
        "What patterns often looked convincing but turned out to be misleading after outcomes were known? "
        "When should I become more confident, and when should I deliberately hold back even if risk factors are present? \n\n"

        "Existing clinician experience (current version to be refined):\n"
        f"{mem}\n\n"

        "Patient-level reasoning outputs for some prior patients:\n"
        f"{joined_reason}\n\n"

        "Corresponding calibration lessons for these patients, derived from true patient outcomes:\n"
        f"{joined}\n\n"


        "Strict constraints:\n"
        "- Write exactly ONE compact paragraph.\n"
        "- Do NOT use lists, numbering, headings, or special formatting in the output.\n"


        "Write the updated clinician experience paragraph now."
    )


def memory_refresh_prompt_qwen(
    new_calib_paragraphs: list[str],
    new_reason_paragraphs: list[str],
    memory: Dict[str, Any],
) -> str:
    """
    Refresh experience_memory.calibration_lessons using recent calibration paragraphs.
    Output: ONE short paragraph (no structure) that will become the single lesson stored.
    """
    reason_cards = [i for i in new_reason_paragraphs]
    calib_cards = [i for i in new_calib_paragraphs]


    joined_reason = "\n\n".join(reason_cards)
    joined = "\n\n".join(calib_cards)
    mem = _memory_compact(memory)


    return (
        "You are a physician specializing in neurodegenerative diseases who has evaluated many patients using EHR data "
        f"and has repeatedly calibrated your clinical judgments against true idiopathic {fullname} ({shortname}) outcomes. "
        "Over time, you have accumulated experience about which clinical patterns tend to be reliable and which tend to mislead. "
        "Your task now is to maintain and refine a clinician experience summary that represents how an experienced doctor "
        "internally adjusts intuition after seeing many outcomes.\n\n"

        "This clinician experience is NOT a case summary. "
        "It is a distilled set of clinical instincts, for example, that guide how evidence should be weighted, how confidence should be calibrated, "
        f"and how common reasoning traps should be avoided when judging whether a patient has {shortname} at the current visit.\n\n"

        "You should think like a senior doctor asking:\n"
        "E.g., across many patients, what patterns consistently deserve more trust? "
        "What patterns often looked convincing but turned out to be misleading after outcomes were known? "
        "When should I become more confident, and when should I deliberately hold back even if risk factors are present? \n\n"

        "Existing clinician experience (current version to be refined):\n"
        f"{mem}\n\n"

        "Patient-level reasoning outputs for some prior patients:\n"
        f"{joined_reason}\n\n"

        "Corresponding calibration lessons for these patients, derived from true patient outcomes:\n"
        f"{joined}\n\n"


        "Strict constraints:\n"
        "- Write exactly ONE compact paragraph.\n"
        "- Do NOT use lists, numbering, headings, or special formatting in the output.\n"


        "Write the updated clinician experience paragraph now."
    )


