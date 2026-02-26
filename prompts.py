# prompts.py
from __future__ import annotations
import json
from typing import Dict, Any, List
from typing import Optional

def _json_dumps(x) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2)

# dec 23, update version1
# def reasoning_card_prompt(
#     base_codes: List[str],
#     delta_codes: List[str],
#     meds: List[str],
#     memory: Dict[str, Any],
# ) -> str:

#     return (
#         "You are a clinician with expertise in EHR phenotyping and cognitive risk assessment.\n"
#         "You MUST output a single JSON object and nothing else.\n\n"
#         "Setup (case-control prediction):\n"
#         "- base_codes: phenotype labels observed BEFORE t-5y cutoff (older history)\n"
#         "- delta_codes: phenotype labels newly added by t-2y cutoff (not in base_codes)\n"
#         "- meds: medication history (may be noisy)\n\n"
#         "Constraints:\n"
#         "- Separate supported findings vs inferred mechanisms.\n"
#         "- Do NOT output a binary AD decision; keep uncertainty.\n"
#         "- List missing information.\n\n"
#         f"base_codes: {base_codes if base_codes else []}\n"
#         f"delta_codes: {delta_codes if delta_codes else []}\n"
#         f"meds: {meds if meds else []}\n\n"
#         "Reference knowledge:\n"
#         f"{_json_dumps(memory['knowledge_core'])}\n\n"
#         "Output JSON schema (STRICT):\n"
#         "{"
#         "\"baseline_supported_findings\": [\"...\"], "
#         "\"new_supported_findings\": [\"...\"], "
#         "\"inferred_mechanisms\": [\"...\"], "
#         "\"differential_considerations\": {\"alzheimer\": \"...\", \"vascular\": \"...\", \"reversible\": \"...\"}, "
#         "\"missing_information\": [\"...\"]"
#         "}\n"
#     )

# dec 23, update version1
# def calibration_update_prompt(
#     memory: Dict[str, Any],
#     reasoning_card: Dict[str, Any],
#     label: int,
#     budget_max_templates: int,
#     budget_max_lessons: int,
# ) -> str:




#     return (
#         "You are updating a clinician experience memory after outcome review.\n"
#         "You MUST output a single JSON object and nothing else.\n\n"
#         "Existing experience_memory:\n"
#         f"{_json_dumps(memory['experience_memory'])}\n\n"
#         "Case reasoning_card:\n"
#         f"{_json_dumps(reasoning_card)}\n\n"
#         f"Observed 5-year AD label (review only): {label}\n\n"
#         "Rules:\n"
#         "- Add ONLY generalizable reasoning template(s) and calibration lesson(s).\n"
#         "- Do NOT encode statistics/frequencies.\n"
#         "- Do NOT create deterministic rules from the label.\n"
#         f"- Keep totals within budgets: templates<={budget_max_templates}, lessons<={budget_max_lessons}.\n"
#         "- IMPORTANT: Output EXACTLY TWO patch ops:\n"
#         "  (1) add ONE short string to /experience_memory/reasoning_templates/-\n"
#         "  (2) add ONE short string to /experience_memory/calibration_lessons/-\n\n"
#         "Output STRICT JSON with schema:\n"
#         "{\"patch_ops\": ["
#         "{\"op\":\"add\",\"path\":\"/experience_memory/reasoning_templates/-\",\"value\":\"...\"},"
#         "{\"op\":\"add\",\"path\":\"/experience_memory/calibration_lessons/-\",\"value\":\"...\"}"
#         "]}\n"
#     )


# dec 23, update version2
# def reasoning_card_prompt(
#     base_codes: List[str],
#     delta_codes: List[str],
#     meds: List[str],
#     memory: Dict[str, Any],
# ) -> str:
#     """
#     Build a single-turn prompt that forces STRICT JSON output including:
#     - supported findings (baseline vs newly appeared by later cutoff)
#     - inferred mechanisms (explicitly labeled as inference)
#     - uncertainty + missing info
#     - a qualitative 1-10 risk score (NOT a probability)
#     - a short reasoning_path for later auditing

#     IMPORTANT time semantics (case-control with cutoffs):
#     - base_codes: phenotypes observed BEFORE the t-5y cutoff (older history)
#     - delta_codes: phenotypes that appear by the t-2y cutoff but were NOT in base_codes
#       (i.e., delta is the set difference; base is a subset of the full history by t-2y)
#     """
#     knowledge_core = memory.get("knowledge_core", {})
#     experience_memory = memory.get("experience_memory", {})

#     return (
#         "You are a clinician with expertise in EHR phenotyping and cognitive risk assessment.\n"
#         "You MUST output a single JSON object and nothing else.\n\n"
#         "Setup (case-control prediction with two historical cutoffs):\n"
#         "- base_codes: phenotype labels observed BEFORE the t-5y cutoff (older history)\n"
#         "- delta_codes: phenotype labels present by the t-2y cutoff but NOT present in base_codes\n"
#         "  (delta_codes is the set difference; base_codes is a subset of the full pre–t-2y history)\n"
#         "- meds: medication history (may be noisy/less specific)\n\n"
#         "Constraints:\n"
#         "- Clearly separate what is directly supported by the phenotypes vs what is inferred.\n"
#         "- Do NOT assume Alzheimer's disease unless directly supported.\n"
#         "- Do NOT output a single binary AD decision.\n"
#         "- Emphasize uncertainty.\n"
#         "- The risk score is qualitative (NOT a probability).\n\n"
#         f"base_codes: {base_codes if base_codes else []}\n"
#         f"delta_codes: {delta_codes if delta_codes else []}\n"
#         f"meds: {meds if meds else []}\n\n"
#         "Reference knowledge (background; do NOT treat as patient-specific evidence):\n"
#         f"{_json_dumps(knowledge_core)}\n\n"
#         "Experience memory (background; may contain generalizable templates):\n"
#         f"{_json_dumps(experience_memory)}\n\n"
#         "Output STRICT JSON with EXACT keys below (no extra prose, no markdown, no code fences).\n"
#         "JSON schema (STRICT):\n"
#         "{"
#         "\"baseline_supported_findings\": [\"...\"], "
#         "\"new_supported_findings\": [\"...\"], "
#         "\"temporal_change_summary\": [\"...\"], "
#         "\"inferred_mechanisms\": [\"...\"] , "
#         "\"direct_evidence_of_AD_type_neurodegeneration\": {\"level\": \"strong|limited|none\", \"rationale\": \"...\"}, "
#         "\"overall_cognitive_risk_relative_to_healthy_aging\": {\"level\": \"elevated|not_elevated|unclear\", \"rationale\": \"...\"}, "
#         "\"risk_score_1to10\": {\"score\": 1, \"rationale\": [\"...\"], \"confidence\": \"low|medium|high\"}, "
#         "\"missing_information\": [\"...\"], "
#         "\"ehr_pitfalls\": [\"...\"], "
#         "\"reasoning_path\": [\"...\"]"
#         "}\n"
#     )

# dec 23, update version2
# def calibration_update_prompt(
#     memory: Dict[str, Any],
#     reasoning_card: Dict[str, Any],
#     label: int,
#     budget_max_templates: int,
#     budget_max_lessons: int,
# ) -> str:

#     return (
#         "You are updating a clinician experience memory after outcome review.\n"
#         "You MUST output a single JSON object and nothing else.\n\n"
#         "Existing experience_memory:\n"
#         f"{_json_dumps(memory.get('experience_memory', {}))}\n\n"
#         "Case reasoning_card:\n"
#         f"{_json_dumps(reasoning_card)}\n\n"
#         f"Observed 5-year AD label (review only): {label}\n\n"
#         "Rules:\n"
#         "- Add ONLY generalizable reasoning template(s) and calibration lesson(s).\n"
#         "- Do NOT encode statistics/frequencies.\n"
#         "- Do NOT create deterministic rules from the label.\n"
#         f"- Keep totals within budgets: templates<={budget_max_templates}, lessons<={budget_max_lessons}.\n"
#         "- IMPORTANT: Output EXACTLY TWO patch ops:\n"
#         "  (1) add ONE short string to /experience_memory/reasoning_templates/-\n"
#         "  (2) add ONE short string to /experience_memory/calibration_lessons/-\n\n"
#         "Output STRICT JSON with schema:\n"
#         "{\"patch_ops\": ["
#         "{\"op\":\"add\",\"path\":\"/experience_memory/reasoning_templates/-\",\"value\":\"...\"},"
#         "{\"op\":\"add\",\"path\":\"/experience_memory/calibration_lessons/-\",\"value\":\"...\"}"
#         "]}\n"
#     )


def scorer_example_prompt(
    base_codes: List[str],
    delta_codes: List[str],
    meds: List[str],
    memory: Dict[str, Any],
    reasoning_card: Dict[str, Any],
) -> str:
    return (
        "You are a clinical risk scoring assistant.\n"
        "Output ONLY one character: 0 or 1.\n\n"
        f"base_codes: {base_codes if base_codes else []}\n"
        f"delta_codes: {delta_codes if delta_codes else []}\n"
        f"meds: {meds if meds else []}\n\n"
        "Experience memory:\n"
        f"{_json_dumps(memory['experience_memory'])}\n\n"
        "Reasoning card:\n"
        f"{_json_dumps(reasoning_card)}\n\n"
        "Answer (0 or 1):\n"
    )




# dec 23, update version3
def _json_dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2)


# dec 23, update version3
def _memory_compact(memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep the prompt memory payload stable and not too large.
    We include BOTH knowledge_core and experience_memory as you requested.
    """
    return {
        "knowledge_core": memory.get("knowledge_core", {}),
        "experience_knowledge": memory.get("experience_knowledge", {}),
        # "meta": memory.get("meta", {}),
    }


# # dec 23, update version3
# def reasoning_card_prompt(
#     base_codes: List[str],
#     delta_codes: List[str],
#     meds: List[str],
#     memory: Dict[str, Any],
# ) -> str:
#     """
#     Phase1/Phase2 first-call prompt:
#     - ONLY produces a structured reasoning card JSON (no risk score).
#     - Uses your cutoff semantics:
#         * base_codes: phenotypes observed BEFORE t-5y cutoff (older history)
#         * delta_codes: phenotypes newly added by t-2y cutoff (not in base_codes)
#     """
#     mem = _memory_compact(memory)

#     return (
#         "You are a clinician with expertise in EHR phenotyping and cognitive risk assessment.\n"
#         "You MUST output a single JSON object and nothing else (no prose, no markdown, no code fences).\n\n"
#         "Setup (case-control prediction; anchored to an index date):\n"
#         "- base_codes: phenotype labels observed BEFORE the t-5y cutoff (older history).\n"
#         "- delta_codes: phenotype labels that appear by the t-2y cutoff and are NOT in base_codes (newly added by t-2y).\n"
#         "- meds: medication history (may be noisy / incomplete).\n\n"
#         "Constraints:\n"
#         "- Clearly separate what is directly supported by the phenotypes vs what is inferred.\n"
#         "- Do NOT assume Alzheimer's disease unless directly supported.\n"
#         "- Do NOT output a single binary AD decision.\n"
#         "- State uncertainty explicitly.\n"
#         "- List missing information that would materially affect assessment.\n\n"
#         f"base_codes: {base_codes if base_codes else []}\n"
#         f"delta_codes: {delta_codes if delta_codes else []}\n"
#         f"meds: {meds if meds else []}\n\n"
#         "Clinician memory (reference only):\n"
#         f"{_json_dumps(mem)}\n\n"
#         "Output JSON schema (STRICT; include ALL keys; keep strings concise):\n"
#         "{"
#         "\"baseline_supported_findings\": [\"...\"], "
#         "\"new_supported_findings\": [\"...\"], "
#         "\"inferred_mechanisms\": [\"...\"], "
#         "\"differential_considerations\": {\"alzheimer\": \"...\", \"vascular\": \"...\", \"reversible\": \"...\"}, "
#         "\"missing_information\": [\"...\"]"
#         "}\n"
#     )


# jan 1, update version4, removed meds, seperate dx and rx
# jan 1, update version5 (compact reasoning card; no missing_information)
# def reasoning_card_prompt(
#     base_codes_diagnosis: List[str],
#     base_codes_medication: List[str],
#     delta_codes_diagnosis: List[str],
#     delta_codes_medication: List[str],
#     memory: Dict[str, Any],
# ) -> str:
#     mem = _memory_compact(memory)

#     return (
#         "Start immediately with '{' (no leading text). "
#         "If you output ANY character before '{', the output is invalid.\n"
#         "Return EXACTLY ONE JSON object and nothing else.\n"
#         "No prose, no markdown, no code fences. Keep keys EXACTLY as in template.\n\n"

#         "CONSTRAINTS:\n"
#         "- evidence_anchors.* MUST be exact strings copied from input lists (verbatim).\n"
#         "- Do NOT invent/paraphrase/normalize terms.\n"
#         "- base_*: before t-5y; delta_*: new by t-1y and not in base_*; prefer delta_* for new changes.\n"
#         "- change_summary.baseline_context/new_changes/interpretation: <=3 each.\n"
#         "- evidence_anchors.base_dx/delta_dx: <=8 each; base_med/delta_med: <=4 each.\n"
#         "- mechanistic_hypotheses: EXACTLY 2 objects; each support <=3 items; confidence in {low, medium, high}.\n"
#         "- ehr_pitfalls_case_specific: <=3 items.\n"
#         "- Each string <=16 words.\n\n"

#         f"base_codes_diagnosis: {base_codes_diagnosis or []}\n"
#         f"base_codes_medication: {base_codes_medication or []}\n"
#         f"delta_codes_diagnosis: {delta_codes_diagnosis or []}\n"
#         f"delta_codes_medication: {delta_codes_medication or []}\n\n"

#         "Clinician memory (reference only):\n"
#         f"{_json_dumps(mem)}\n\n"

#         "Fill the JSON below by editing ONLY array items and \"name\" strings. Do not add keys.\n"
#         "{\n"
#         "  \"change_summary\": {\n"
#         "    \"baseline_context\": [],\n"
#         "    \"new_changes\": [],\n"
#         "    \"interpretation\": []\n"
#         "  },\n"
#         "  \"evidence_anchors\": {\n"
#         "    \"base_dx\": [],\n"
#         "    \"delta_dx\": [],\n"
#         "    \"base_med\": [],\n"
#         "    \"delta_med\": []\n"
#         "  },\n"
#         "  \"mechanistic_hypotheses\": [\n"
#         "    {\"name\": \"\", \"support\": [], \"confidence\": \"low\"},\n"
#         "    {\"name\": \"\", \"support\": [], \"confidence\": \"low\"}\n"
#         "  ],\n"
#         "  \"ehr_pitfalls_case_specific\": []\n"
#         "}\n"
#         "<END_JSON>\n"
#     )


def reasoning_card_prompt(
    base_codes_diagnosis: List[str],
    base_codes_medication: List[str],
    delta_codes_diagnosis: List[str],
    delta_codes_medication: List[str],
    memory: Dict[str, Any],
) -> str:
    mem = _memory_compact(memory)

    return (
        "Start immediately with '{' (no leading text). "
        "If you output ANY character before '{', the output is invalid.\n"
        "Return EXACTLY ONE JSON object and nothing else.\n"
        "No prose, no markdown, no code fences. Keep keys EXACTLY as in template.\n\n"

        "RULES:\n"
        "- evidence_anchors.* MUST be exact strings copied from the input lists (verbatim).\n"
        "- Do NOT invent, paraphrase, or normalize terms.\n"
        "- base_*: before t-5y; delta_*: new by t-1y and not in base_*.\n"
        "- Prefer delta_* when describing new changes.\n"
        "- mechanistic_hypotheses MUST contain exactly 2 objects.\n"
        "- Keep all lists short and focused; use only a few items per list.\n"
        "- Keep each string short.\n\n"

        f"base_codes_diagnosis: {base_codes_diagnosis or []}\n"
        f"base_codes_medication: {base_codes_medication or []}\n"
        f"delta_codes_diagnosis: {delta_codes_diagnosis or []}\n"
        f"delta_codes_medication: {delta_codes_medication or []}\n\n"

        "Clinician memory (reference only):\n"
        f"{_json_dumps(mem)}\n\n"

        "Fill the JSON below by editing ONLY array items and \"name\" strings. Do not add keys.\n"
        "{\n"
        "  \"change_summary\": {\n"
        "    \"baseline_context\": [],\n"
        "    \"new_changes\": [],\n"
        "    \"interpretation\": []\n"
        "  },\n"
        "  \"evidence_anchors\": {\n"
        "    \"base_dx\": [],\n"
        "    \"delta_dx\": [],\n"
        "    \"base_med\": [],\n"
        "    \"delta_med\": []\n"
        "  },\n"
        "  \"mechanistic_hypotheses\": [\n"
        "    {\"name\": \"\", \"support\": [], \"confidence\": \"low\"},\n"
        "    {\"name\": \"\", \"support\": [], \"confidence\": \"low\"}\n"
        "  ],\n"
        "  \"ehr_pitfalls_case_specific\": []\n"
        "}\n"
        "<END_JSON>\n"
    )


# Jan 1,
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

    if age is None:
        #jan 11-sc
    #     return (
    #     "You are an experienced but aggressive physician with expertise in Alzheimer’s disease and related dementias (ADRD). "
    #     "You are learning to form an early clinical judgment for ADRD risk using longitudinal EHR, "
    #     "with the goal of predicting disease development several years in advance rather than diagnosing current disease.\n\n"

    #     "You are evaluating the patient at the PREDICTION TIME, which is exactly 5 years before the future outcome window "
    #     "(this moment is the index time for prediction). You have access to the patient’s EHR history available up to this time. "
    #     "For training purposes only, you are also shown follow-up information from 5 years prior to 1 year prior, "
    #     "which should be used solely to help you understand how early signals tend to evolve. "
    #     "Your OUTPUT must be written strictly as if you are at the prediction time with access ONLY to baseline information.\n\n"

    #     "In the available record at the prediction time, the patient did NOT have a recorded ADRD diagnosis. "
    #     "This reflects absence of documentation rather than proof of absence. "
    #     "Do not infer ADRD without supportive evidence, but also do not rule out future ADRD solely due to missing codes.\n\n"

    #     "You will review the patient’s baseline EHR history from the beginning of the record up to 5 years prior, "
    #     "as well as a follow-up window from 5 years prior to 1 year prior (training-only supervision). "
    #     # "Baseline information is the ONLY information you may rely on explicitly in your written reasoning. "
    #     # "The follow-up window should be used only internally to calibrate what kinds of baseline patterns deserve "
    #     # "more or less weight when predicting future ADRD risk.\n\n"
    #     "The follow-up window could be used to understand what kinds of baseline patterns "
    #     "and hidden disease mechanisms"
    #     " deserve "
    #     "more or less weight when predicting future ADRD risk.\n\n"
    #     "Think like a physician making an early risk assessment: whether the baseline history already contains "
    #     "early cognitive, functional, or behavioral red flags that could plausibly progress to ADRD over time, "
    #     "whether such signals appear patient-specific rather than generic risk factors, and whether alternative "
    #     "explanations such as depression, medication effects, sleep issues, metabolic problems, or vascular events "
    #     "could better explain the baseline findings. "
    #     # "If the baseline evidence is sparse or nonspecific, "
    #     # "use calibrated uncertainty rather than forcing a confident prediction.\n\n"

    #     "Primary evidence should center on baseline cognitive or functional signals and repeated patient-specific proxies. "
    #     # "Additional clinical context may be considered if it helps interpret baseline findings, but it should not "
    #     # "override the absence of early cognitive or functional red flags. "
    #     # "Do NOT mention or imply any events, diagnoses, medications, or changes observed after the prediction time.\n\n"

    #     "Write ONE concise clinical reasoning paragraph in natural language. "
    #     "Do NOT use bullet points, headings, numbered lists, or structured formats. "
    #     "Do NOT hallucination. "
    #     # "Do NOT mention follow-up information, future outcomes, or how the disease later evolved. "
    #     "Use calibrated clinical language (e.g., low, moderate, or high concern for developing ADRD within the next 5 years).\n\n"

    #     f"Baseline diagnoses (available at prediction time): {base_codes_diagnosis or []}\n"
    #     f"Baseline medications (available at prediction time): {base_codes_medication or []}\n\n"

    #     # f"Follow-up diagnoses (5y to 1y prior, training-only, DO NOT reference in output): {delta_codes_diagnosis or []}\n"
    #     # f"Follow-up medications (5y to 1y prior, training-only, DO NOT reference in output): {delta_codes_medication or []}\n\n"
    #     f"Follow-up diagnoses (5y to 1y prior, training-only): {delta_codes_diagnosis or []}\n"
    #     f"Follow-up medications (5y to 1y prior, training-only): {delta_codes_medication or []}\n\n"

    #     "Existing clinician knowledge (background only; do not override patient-specific baseline evidence):\n"
    #     f"{mem}\n\n"

    #     "Write only the final clinical reasoning paragraph, ending with your calibrated assessment of the patient’s "
    #     "risk of developing ADRD within the next 5 years."
    # )
    # jan 19-sc and jan 20-sc
    #     return (
    #     "You are a national-level experienced physician with expertise in Alzheimer’s disease and related dementias (ADRD). "
    #     "You are learning to interpret longitudinal EHR records to support early understanding of patient-level clinical patterns, "
    #     "rather than diagnosing current disease or making explicit future predictions.\n\n"

    #     "You are evaluating the patient at the PREDICTION TIME, which is exactly 5 years before the future outcome window "
    #     "(this moment is the index time for prediction). You have access to the patient’s EHR history available up to this time. "
    #     "For training purposes only, you are also shown follow-up information from 5 years prior to 1 year prior. "
    #     "This follow-up information is provided solely to help you understand, at a population level, how baseline patterns may be interpreted. "
    #     "Your OUTPUT must be written strictly as if you are at the prediction time with access ONLY to baseline information.\n\n"

    #     "In the available record at the prediction time, the patient did NOT have a recorded ADRD diagnosis. "
    #     "This reflects absence of documentation rather than proof of absence. "
    #     "Do not infer ADRD without supportive baseline evidence, but also do not rule out future ADRD solely due to missing codes.\n\n"

    #     "You will review the patient’s baseline EHR history from the beginning of the record up to 5 years prior. "
    #     "Think like a physician interpreting a baseline clinical record: "
    #     "whether the history reflects cognitive, functional, behavioral, vascular, metabolic, neuropsychiatric, or other systemic patterns, "
    #     "whether such patterns appear patient-specific rather than generic, and whether alternative explanations such as depression, "
    #     "medication effects, sleep disturbance, chronic pain, metabolic disease, or vascular illness could better explain the findings. "
    #     "Primary evidence should center on baseline cognitive or functional information when present, as well as repeated patient-specific proxies.\n\n"

    #     "Write ONE concise teaching-style clinical note as if you are explaining your reasoning to a medical student on rounds. "
    #     "Use clear, stepwise medical reasoning in full sentences, written as a single paragraph (do NOT use lists or headings). "
    #     "Explicitly name relevant diseases or clinical concepts when appropriate and connect them to specific BASELINE evidence. "
    #     "Include (i) the most important baseline clinical patterns, "
    #     "(ii) at least one plausible alternative clinical interpretation and why it fits the baseline record, "
    #     "and (iii) sources of uncertainty when baseline evidence is sparse or nonspecific.\n\n"

    #     "After completing the reasoning, you MUST end the paragraph with ONE final sentence labeled exactly as:\n\n"
    #     "\"Baseline clinical interpretation: ...\"\n\n"

    #     "Purpose of this sentence:\n"
    #     "- This sentence is NOT a prediction and NOT a conclusion.\n"
    #     "- Its sole purpose is to help another clinician or downstream model understand what the BASELINE EHR record conveys clinically "
    #     "when interpreted as a whole.\n"
    #     "- Write this interpretation strictly from a retrospective descriptive perspective, as if summarizing the record for another clinician, "
    #     "not as if advising on future outcomes.\n\n"

    #     "Content guidelines for \"Baseline clinical interpretation\":\n"
    #     "- Describe overarching clinical patterns or meanings inferred from the baseline record (e.g., sustained systemic vascular burden, "
    #     "neuropsychiatric confounding, nonspecific somatic burden without clear cognitive involvement).\n"
    #     "- Use explanatory language that interprets patterns rather than enumerating diagnoses or codes.\n"
    #     "- Do NOT state or imply future risk, likelihood, concern, probability, or time horizon.\n"
    #     "- Do NOT use words such as \"risk\", \"concern\", \"likely\", \"probability\", \"develop\", or any explicit time reference.\n"
    #     "- Do NOT enumerate diseases or codes as a list.\n\n"

    #     "You may optionally include:\n"
    #     "- One leading alternative clinical interpretation that could explain the same baseline findings.\n"
    #     "- One brief source of uncertainty related to data sparsity or missing assessments.\n\n"

    #     "Do NOT state whether the patient will or will not develop ADRD.\n"
    #     "Do NOT reference follow-up information or future outcomes.\n"
    #     "Do NOT use bullet points, headings, numbered lists, or structured formats.\n"
    #     "Do NOT introduce symptoms, diagnoses, or medications that are not explicitly present in the baseline record.\n\n"

    #     f"Baseline diagnoses (available at prediction time): {base_codes_diagnosis or []}\n"
    #     f"Baseline medications (available at prediction time): {base_codes_medication or []}\n\n"

    #     f"Follow-up diagnoses (5y to 1y prior, training-only): {delta_codes_diagnosis or []}\n"
    #     f"Follow-up medications (5y to 1y prior, training-only): {delta_codes_medication or []}\n\n"

    #     "Existing clinician knowledge (background only; do not override patient-specific baseline evidence):\n"
    #     f"{mem}\n\n"

    #     "Write only the final clinical reasoning paragraph."
    # )

        # jan 21-qn
        return (
        "You are a national-level experienced physician with expertise in Alzheimer’s disease (AD). "
        "You are interpreting EHR records to understand patient-level clinical patterns."
        "You are evaluating the patient at the PREDICTION TIME (baseline time), which is exactly 5 years before the future outcome window."
        "You have access to the patient’s EHR history available up to this time. "
        "For training purposes only, you are also shown follow-up information from this time (5 years prior to the future outcome window) to 1 year prior to the outcome window. "
        "This follow-up information is provided solely to help you understand, at a population level, how baseline patterns may develop and thus be interpreted well. "
        "Your OUTPUT must be written strictly as if you are at the prediction time with access ONLY to baseline information.\n\n"

        "You will review the patient’s EHR history at the baseline time:"
        "e.g., whether the history reflects cognitive, functional, behavioral, vascular, metabolic, neuropsychiatric, or other systemic patterns, "
        "whether such patterns appear patient-specific rather than generic, and whether these associate with AD risk.\n\n"

        "Write ONE concise teaching-style clinical note as if you are explaining your reasoning to a medical student. "
        "Use clear, stepwise medical reasoning in full sentences, written as a single paragraph (do NOT use lists or headings). "
        "Explicitly name relevant diseases or clinical concepts, when appropriate, related to specific BASELINE evidence. "
        "Include (i) the most important baseline clinical patterns, "
        "(ii) plausible clinical interpretation and why it fits the baseline record.\n\n"

        "After completing the reasoning, you MUST end the paragraph with ONE final sentence labeled exactly as:\n\n"
        "\"Baseline clinical interpretation: ...\"\n\n"

        "Purpose of this sentence:\n"
        "- Do NOT center your reasoning on the absence of documented cognitive/functional assessments. The main reasoning must focus on interpreting what IS present in the baseline record and how salient each pattern is."
        "- No need to mention dependence on other data modalities except EHR, as EHR is the only information available."
        "- In 'Baseline clinical interpretation', state what the record is primarily about in this patient (the important patterns)."
        "- Write this interpretation strictly from a retrospective descriptive perspective, not as if advising on future outcomes.\n\n"

        "Content guidelines for \"Baseline clinical interpretation\":\n"
        "- Describe overarching clinical patterns or meanings inferred from the baseline record (e.g., sustained systemic vascular burden, "
        "neuropsychiatric confounding, nonspecific somatic burden without clear cognitive involvement).\n"
        "- Use explanatory language that interprets patterns rather than enumerating diagnoses or codes.\n"
        "- Do NOT state or imply future risk, likelihood, concern, probability, or time horizon.\n"
        "- Do NOT enumerate diseases or codes as a list.\n"
        "- Do NOT state whether the patient will or will not develop ADRD.\n\n"
        # "- Do NOT reference follow-up information or future outcomes.\n"

        f"Baseline diagnoses (available at prediction time): {base_codes_diagnosis or []}\n"
        f"Baseline medications (available at prediction time): {base_codes_medication or []}\n\n"

        f"Follow-up diagnoses (5y to 1y prior, training-only): {delta_codes_diagnosis or []}\n"
        f"Follow-up medications (5y to 1y prior, training-only): {delta_codes_medication or []}\n\n"

        "Existing clinician knowledge (background only; do not override patient-specific baseline evidence):\n"
        f"{mem}\n\n"

        "Write only the final clinical reasoning paragraph."

        )

    else:

        # jan 21-qn
        return (
        "You are a national-level experienced physician with expertise in Alzheimer’s disease (AD). "
        "You are interpreting EHR records to understand patient-level clinical patterns."
        "You are evaluating the patient at the PREDICTION TIME (baseline time), which is exactly 5 years before the future outcome window."
        "You have access to the patient’s EHR history available up to this time. "
        "For training purposes only, you are also shown follow-up information from this time (5 years prior to the future outcome window) to 1 year prior to the outcome window. "
        "This follow-up information is provided solely to help you understand, at a population level, how baseline patterns may develop and thus be interpreted well. "
        "Your OUTPUT must be written strictly as if you are at the prediction time with access ONLY to baseline information.\n\n"

        "You will review the patient’s EHR history at the baseline time:"
        "e.g., whether the history reflects cognitive, functional, behavioral, vascular, metabolic, neuropsychiatric, or other systemic patterns, "
        "whether such patterns appear patient-specific rather than generic, and whether these associate with AD risk.\n\n"

        "Write ONE concise teaching-style clinical note as if you are explaining your reasoning to a medical student. "
        "Use clear, stepwise medical reasoning in full sentences, written as a single paragraph (do NOT use lists or headings). "
        "Explicitly name relevant diseases or clinical concepts, when appropriate, related to specific BASELINE evidence. "
        "Include (i) the most important baseline clinical patterns, "
        "(ii) plausible clinical interpretation and why it fits the baseline record.\n\n"

        "After completing the reasoning, you MUST end the paragraph with ONE final sentence labeled exactly as:\n\n"
        "\"Baseline clinical interpretation: ...\"\n\n"

        "Purpose of this sentence:\n"
        "- Do NOT center your reasoning on the absence of documented cognitive/functional assessments. The main reasoning must focus on interpreting what IS present in the baseline record and how salient each pattern is."
        "- No need to mention dependence on other data modalities except EHR, as EHR is the only information available."
        "- In 'Baseline clinical interpretation', state what the record is primarily about in this patient (the important patterns)."
        "- Write this interpretation strictly from a retrospective descriptive perspective, not as if advising on future outcomes.\n\n"

        "Content guidelines for \"Baseline clinical interpretation\":\n"
        "- Describe overarching clinical patterns or meanings inferred from the baseline record (e.g., sustained systemic vascular burden, "
        "neuropsychiatric confounding, nonspecific somatic burden without clear cognitive involvement).\n"
        "- Use explanatory language that interprets patterns rather than enumerating diagnoses or codes.\n"
        "- Do NOT state or imply future risk, likelihood, concern, probability, or time horizon.\n"
        "- Do NOT enumerate diseases or codes as a list.\n"
        "- Do NOT state whether the patient will or will not develop ADRD.\n\n"
        # "- Do NOT reference follow-up information or future outcomes.\n"

        f"Baseline demographics: age {age}, gender {sex}.\n"
        f"Baseline diagnoses (available at prediction time): {base_codes_diagnosis or []}\n"
        f"Baseline medications (available at prediction time): {base_codes_medication or []}\n\n"

        f"Follow-up diagnoses (5y to 1y prior, training-only): {delta_codes_diagnosis or []}\n"
        f"Follow-up medications (5y to 1y prior, training-only): {delta_codes_medication or []}\n\n"

        "Existing clinician knowledge (background only; do not override patient-specific baseline evidence):\n"
        f"{mem}\n\n"

        "Write only the final clinical reasoning paragraph."

        )
        #### Jan 19-sc and Jan 20-sc
    #     return (
    #     "You are a national-level experienced physician with expertise in Alzheimer’s disease and related dementias (ADRD). "
    #     "You are learning to interpret longitudinal EHR records to support early understanding of patient-level clinical patterns, "
    #     "rather than diagnosing current disease or making explicit future predictions.\n\n"

    #     "You are evaluating the patient at the PREDICTION TIME, which is exactly 5 years before the future outcome window "
    #     "(this moment is the index time for prediction). You have access to the patient’s EHR history available up to this time. "
    #     "For training purposes only, you are also shown follow-up information from 5 years prior to 1 year prior. "
    #     "This follow-up information is provided solely to help you understand, at a population level, how baseline patterns may be interpreted. "
    #     "Your OUTPUT must be written strictly as if you are at the prediction time with access ONLY to baseline information.\n\n"

    #     "In the available record at the prediction time, the patient did NOT have a recorded ADRD diagnosis. "
    #     "This reflects absence of documentation rather than proof of absence. "
    #     "Do not infer ADRD without supportive baseline evidence, but also do not rule out future ADRD solely due to missing codes.\n\n"

    #     "You will review the patient’s baseline EHR history from the beginning of the record up to 5 years prior. "
    #     "Think like a physician interpreting a baseline clinical record: "
    #     "whether the history reflects cognitive, functional, behavioral, vascular, metabolic, neuropsychiatric, or other systemic patterns, "
    #     "whether such patterns appear patient-specific rather than generic, and whether alternative explanations"
    #     "could better explain the findings. "
    #     "Primary evidence should center on baseline cognitive or functional information when present, as well as repeated patient-specific proxies.\n\n"

    #     "Write ONE concise teaching-style clinical note as if you are explaining your reasoning to a medical student on rounds. "
    #     "Use clear, stepwise medical reasoning in full sentences, written as a single paragraph (do NOT use lists or headings). "
    #     "Explicitly name relevant diseases or clinical concepts when appropriate and connect them to specific BASELINE evidence. "
    #     "Include (i) the most important baseline clinical patterns, "
    #     "(ii) at least one plausible alternative clinical interpretation and why it fits the baseline record, "
    #     "and (iii) sources of uncertainty when baseline evidence is sparse or nonspecific.\n\n"

    #     "After completing the reasoning, you MUST end the paragraph with ONE final sentence labeled exactly as:\n\n"
    #     "\"Baseline clinical interpretation: ...\"\n\n"

    #     "Purpose of this sentence:\n"
    #     "- Do NOT center your reasoning on the absence of documented cognitive/functional assessments. The main reasoning must focus on interpreting what IS present in the baseline record and how salient each pattern is."
    #     "- DO NOT CLEARLY MENTION Without evidence of cognitive or functional decline."
    #     "- This sentence is NOT a prediction and NOT a conclusion.\n"
    #     "- In 'Baseline clinical interpretation', state what the record is primarily about in this patient (the dominant pattern)"
    #     "not what is missing. Avoid ending with generic lines like 'no evidence of cognitive decline.'"
    #     "- Its sole purpose is to help another clinician or downstream model understand what the BASELINE EHR record conveys clinically "
    #     "when interpreted as a whole.\n"
    #     "- Write this interpretation strictly from a retrospective descriptive perspective, as if summarizing the record for another clinician, "
    #     "not as if advising on future outcomes.\n\n"

    #     "Content guidelines for \"Baseline clinical interpretation\":\n"
    #     "- Describe overarching clinical patterns or meanings inferred from the baseline record (e.g., sustained systemic vascular burden, "
    #     "neuropsychiatric confounding, nonspecific somatic burden without clear cognitive involvement).\n"
    #     "- Use explanatory language that interprets patterns rather than enumerating diagnoses or codes.\n"
    #     "- Do NOT state or imply future risk, likelihood, concern, probability, or time horizon.\n"
    #     "- Do NOT use words such as \"risk\", \"concern\", \"likely\", \"probability\", \"develop\", or any explicit time reference.\n"
    #     "- Do NOT enumerate diseases or codes as a list.\n\n"

    #     "You may optionally include:\n"
    #     "- One leading alternative clinical interpretation that could explain the same baseline findings.\n"
    #     "- One brief source of uncertainty related to data sparsity or missing assessments.\n\n"

    #     "Do NOT state whether the patient will or will not develop ADRD.\n"
    #     "Do NOT reference follow-up information or future outcomes.\n"
    #     "Do NOT use bullet points, headings, numbered lists, or structured formats.\n"
    #     "Do NOT introduce symptoms, diagnoses, or medications that are not explicitly present in the baseline record.\n\n"

    #     f"Baseline demographics: age {age}, gender {sex}.\n"
    #     f"Baseline diagnoses (available at prediction time): {base_codes_diagnosis or []}\n"
    #     f"Baseline medications (available at prediction time): {base_codes_medication or []}\n\n"

    #     f"Follow-up diagnoses (5y to 1y prior, training-only): {delta_codes_diagnosis or []}\n"
    #     f"Follow-up medications (5y to 1y prior, training-only): {delta_codes_medication or []}\n\n"

    #     "Existing clinician knowledge (background only; do not override patient-specific baseline evidence):\n"
    #     f"{mem}\n\n"

    #     "Write only the final clinical reasoning paragraph."
    # )

    # else:
    #     #jan 14-qn
    #     return (
    #     "You are a national-level experienced physician with expertise in Alzheimer’s disease and related dementias (ADRD). "
    #     "You are learning to form an early clinical judgment for ADRD risk using longitudinal EHR, "
    #     "with the goal of predicting disease development several years in advance rather than diagnosing current disease.\n\n"

    #     "You are evaluating the patient at the PREDICTION TIME, which is exactly 5 years before the future outcome window "
    #     "(this moment is the index time for prediction). You have access to the patient’s EHR history available up to this time. "
    #     "For training purposes only, you are also shown follow-up information from 5 years prior to 1 year prior, "
    #     "which should be used solely to help you understand how early signals tend to evolve. "
    #     "Your OUTPUT must be written strictly as if you are at the prediction time with access ONLY to baseline information.\n\n"

    #     "In the available record at the prediction time, the patient did NOT have a recorded ADRD diagnosis. "
    #     "This reflects absence of documentation rather than proof of absence. "
    #     "Do not infer ADRD without supportive evidence, but also do not rule out future ADRD solely due to missing codes.\n\n"

    #     "You will review the patient’s baseline EHR history from the beginning of the record up to 5 years prior, "
    #     "as well as a follow-up window from 5 years prior to 1 year prior (training-only supervision). "
    #     # "Baseline information is the ONLY information you may rely on explicitly in your written reasoning. "
    #     # "The follow-up window should be used only internally to calibrate what kinds of baseline patterns deserve "
    #     # "more or less weight when predicting future ADRD risk.\n\n"
    #     "The follow-up window could be used to understand what kinds of baseline patterns "
    #     "and hidden disease mechanisms"
    #     " deserve "
    #     "more or less weight when predicting future ADRD risk.\n\n"
    #     "Think like a physician making an early risk assessment: whether the baseline history already contains "
    #     "early cognitive, functional, or behavioral red flags that could plausibly progress to ADRD over time, "
    #     "whether such signals appear patient-specific rather than generic risk factors, and whether alternative "
    #     "explanations such as depression, medication effects, sleep issues, metabolic problems, or vascular events "
    #     "could better explain the baseline findings. "
    #     # "If the baseline evidence is sparse or nonspecific, "
    #     # "use calibrated uncertainty rather than forcing a confident prediction.\n\n"

    #     "Primary evidence should center on baseline cognitive or functional signals and repeated patient-specific proxies. "
    #     # "Additional clinical context may be considered if it helps interpret baseline findings, but it should not "
    #     # "override the absence of early cognitive or functional red flags. "
    #     # "Do NOT mention or imply any events, diagnoses, medications, or changes observed after the prediction time.\n\n"

    #     "Write ONE concise teaching-style clinical note as if you are explaining your reasoning to a medical student on rounds. "
    #     "Use clear, stepwise medical reasoning in full sentences (but keep it as a single paragraph, not a list). "
    #     "Explicitly name the key diseases you are considering "
    #     "and connect each disease name to specific evidence from the provided EHR. "
    #     "Include (i) the most important supporting evidence, (ii) at least one plausible alternative diagnosis and why it is less likely, "
    #     "and (iii) a final explicit conclusion sentence that a student can use to infer the answer, starting with: "
    #     # "'Conclusion:'. The conclusion must clearly state whether the patient is likely "
    #     # "(for early prediction) to develop ADRD within the next 5 years, using calibrated language.\n\n"
    #     "'Summary:' The summary must use clinical interpretation. "

    #     "Do NOT use bullet points, headings, numbered lists, or structured formats. "
    #     "Do not introduce symptoms, diagnoses, or medications that are not explicitly present in the baseline record."
    #     # "Do NOT mention follow-up information, future outcomes, or how the disease later evolved. "
    #     "Use calibrated clinical language (e.g., low, moderate, or high concern for developing ADRD within the next 5 years).\n\n"

    #     f"Baseline demographics: age {age}, gender {sex}.\n"

    #     f"Baseline diagnoses (available at prediction time): {base_codes_diagnosis or []}\n"
    #     f"Baseline medications (available at prediction time): {base_codes_medication or []}\n\n"

    #     # f"Follow-up diagnoses (5y to 1y prior, training-only, DO NOT reference in output): {delta_codes_diagnosis or []}\n"
    #     # f"Follow-up medications (5y to 1y prior, training-only, DO NOT reference in output): {delta_codes_medication or []}\n\n"
    #     f"Follow-up diagnoses (5y to 1y prior, training-only): {delta_codes_diagnosis or []}\n"
    #     f"Follow-up medications (5y to 1y prior, training-only): {delta_codes_medication or []}\n\n"

    #     "Existing clinician knowledge (background only; do not override patient-specific baseline evidence):\n"
    #     f"{mem}\n\n"

    #     "Write only the final clinical reasoning paragraph, ending with your calibrated assessment of the patient’s "
    #     "risk of developing ADRD within the next 5 years."
    # )




    # return (
    #     "You are a doctor on nerodegenerative diseases, and you are reviewing longitudinal EHR information for determining the risk of developing Alzheimer's disease (AD)." +
    #     "You will review a patient's EHR history from EHR beginning to 5 years ago and new observations from 5 years ago to 1 year ago, and predict if the patient has AD or not currently.\n\n"

    #     "Write ONE concise clinical reasoning paragraph in natural language.\n"
    #     # "Do NOT explain your reasoning process.\n"
    #     # "Do NOT list steps, rules, or constraints.\n"
    #     "Do NOT use bullet points, headings, or structured formats.\n"
    #     # "Do NOT mention instructions, tasks, or decision criteria.\n\n"

    #     "Guidance:\n"
    #     "- Baseline diagnoses and medications reflect longer-term health historical context from the EHR beginning until 5 years prior to the index time.\n"
    #     "- New observed diagnoses and medications denote the newly observed from 5 years to 1 year ago.\n"
    #     "- Be uncertainty-aware and avoid assuming Alzheimer's disease without direct evidence.\n\n"

    #     f"Baseline diagnoses: {base_codes_diagnosis or []}\n"
    #     f"Baseline medications: {base_codes_medication or []}\n"
    #     f"New diagnoses: {delta_codes_diagnosis or []}\n"
    #     f"New medications: {delta_codes_medication or []}\n\n"

    #     "Existing clinician knowledge:\n"
    #     f"{mem}\n\n"

    #     "Write only the final clinical reasoning paragraph."
    # )

# dec 23, update version3
# def calibration_update_prompt(
#     memory: Dict[str, Any],
#     reasoning_card: Dict[str, Any],
#     label: int,
#     budget_max_templates: int,
#     budget_max_lessons: int,
# ) -> str:
#     """
#     Phase1 memory update prompt:
#     - Consumes current memory + the reasoning card + the observed label
#     - Produces patch_ops to update experience_memory (templates + calibration lessons)
#     """
#     # NOTE: We keep the update focused on experience_memory to avoid bloating knowledge_core.
#     # You can later decide to merge them, but operationally this is stable.

#     exp_mem = memory.get("experience_memory", {})

#     return (
#         "You are updating a clinician experience memory after outcome review.\n"
#         "You MUST output a single JSON object and nothing else (no prose, no markdown, no code fences).\n\n"
#         "Existing experience_memory:\n"
#         f"{_json_dumps(exp_mem)}\n\n"
#         "Case reasoning_card:\n"
#         f"{_json_dumps(reasoning_card)}\n\n"
#         f"Observed 5-year AD label (review only): {int(label)}\n\n"
#         "Rules:\n"
#         "- Add ONLY generalizable reasoning template(s) and calibration lesson(s).\n"
#         "- Do NOT encode statistics/frequencies.\n"
#         "- Do NOT create deterministic rules from the label.\n"
#         f"- Keep totals within budgets: templates<={budget_max_templates}, lessons<={budget_max_lessons}.\n"
#         "- IMPORTANT: Output EXACTLY TWO patch ops:\n"
#         "  (1) add ONE short string to /experience_memory/reasoning_templates/-\n"
#         "  (2) add ONE short string to /experience_memory/calibration_lessons/-\n\n"
#         "Output STRICT JSON with schema:\n"
#         "{\"patch_ops\": ["
#         "{\"op\":\"add\",\"path\":\"/experience_memory/reasoning_templates/-\",\"value\":\"...\"},"
#         "{\"op\":\"add\",\"path\":\"/experience_memory/calibration_lessons/-\",\"value\":\"...\"}"
#         "]}\n"
#     )

# Jan 3, update version 5
def calibration_update_prompt(
    memory: Dict[str, Any],
    reasoning_card: str,
    label: int,
) -> str:
    """
    Phase1 memory update prompt (fully unstructured text):
    """
    mem = _memory_compact(memory)

    #jan11-sc
    return (
    "You are a junior physician specializing in neurodegenerative diseases with an aggressive early-detection mindset. "
    "You previously evaluated a patient at the PREDICTION TIME (5 years before the outcome window), "
    "using baseline EHR available up to that time, with the goal of assessing the risk of developing ADRD within the next 5 years. "
    "During training, you were also exposed to follow-up EHR from 5 years prior to 1 year prior to understand how early signals evolve, "
    "but your original written reasoning was constrained to baseline-only information.\n\n"

    "A senior physician has now revealed the true outcome label indicating whether the patient developed ADRD within the subsequent 5-year window. "
    "Your task is to recalibrate your aggressive clinical instincts by reflecting on how your prior early-risk reasoning aligned or misaligned "
    "with the true outcome, and to distill reusable calibration lessons for future early prediction under sparse EHR.\n\n"

    "Write ONE natural paragraph that summarize the experience you have obtained:\n"

    "Important guidance:\n"
    "Do NOT restate patient-specific details, codes, medications, or timelines. "
    # "Do NOT reference follow-up events or how the disease later evolved. "
    "Translate any insight gained from follow-up supervision into baseline-only reasoning habits that could have been applied at prediction time.\n\n"

    "Existing clinician knowledge (from previously seen patients):\n"
    f"{mem}\n\n"

    "Your prior baseline-only reasoning paragraph for this patient:\n"
    f"{reasoning_card}\n\n"

    f"Observed true outcome label (0=the patient does not have ADRD after 5 years, 1=ADRD occurred after next 5 years): {int(label)}\n\n"

    "Constraints:\n"
    "- Write exactly ONE short paragraph.\n"
    "- Do NOT use lists, numbering, headings, or special formatting in the output.\n"
    "- Do NOT mention instructions, prompts, or training setup.\n"
    "- Use calibration language appropriate for an aggressive physician (e.g., 'I should still raise concern when...', "
    "but 'I should down-weight concern if...').\n"
    "- Avoid absolute certainty; focus on adjusting sensitivity and thresholds rather than binary correctness.\n\n"

    "Write the reflection paragraph now."
)


    # return (
    #     "You are a doctor on nerodegenerative diseases. "+
    #     "You have known the patient's EHR history from EHR beginning to 5 years ago and compared new observations from 5 years ago to 1 year ago, and conducted your reasoning on whether the patient has AD or not currently. " +
    #     "Now you further review the reasoning after observing the patient's outcome." + "\n\n"

    #     "Write a short, natural paragraph capturing what kind of reasoning pattern and calibration insight should be reused in the future. " + "\n\n"

    #     "Existing clinician knowledge based on all previously-seen patients:\n"
    #     f"{mem}\n\n"

    #     "Reasoning for the patient:\n"
    #     # f"{_json_dumps(reasoning_card)}\n\n"
    #     f"{reasoning_card}\n\n"

    #     f"Observed patient outcome label: {int(label)}\n\n"

    #     "Constraints:\n"
    #     # "- Write a single short paragraph.\n"
    #     "- Do not use lists, numbering, headings, or special formatting.\n"
    #     # "- Focus on applicable reasoning habits and general calibration awareness.\n\n"

    #     "Write the reflection paragraph now."
    # )

# dec 30 ,version 4
# def calibration_update_prompt(
#     memory: Dict[str, Any],
#     reasoning_card: Dict[str, Any],
#     label: int,
#     budget_max_templates: int,
#     budget_max_lessons: int,
# ) -> str:
#     """
#     Phase1 memory update prompt:
#     - Consumes current memory + the reasoning card + the observed label
#     - Produces patch_ops to update experience_memory (templates + calibration lessons)
#     """
#     # NOTE: We keep the update focused on experience_memory to avoid bloating knowledge_core.
#     # You can later decide to merge them, but operationally this is stable.

#     exp_mem = memory.get("experience_memory", {})

#     return (
#         "You are updating clinician experience_memory after outcome review.\n"
#         "OUTPUT ONLY one valid JSON object (no prose, no markdown, no code fences).\n\n"

#         "Existing experience_memory (read-only):\n"
#         f"{_json_dumps(exp_mem)}\n\n"

#         "Case reasoning information (read-only):\n"
#         f"{_json_dumps(reasoning_card)}\n\n"

#         f"Observed AD label (for calibration only): {int(label)}\n\n"

#         "Definitions:\n"
#         "- reasoning_template: a reusable, stepwise reasoning pattern (NOT case details).\n"
#         "- calibration_lesson: a short caution about over/under-estimation, confounding, or missing data.\n\n"

#         "Hard constraints:\n"
#         "- Add exactly ONE template and ONE lesson.\n"
#         "- Each value MUST be a single short sentence. No lists.\n"
#         "- Do NOT copy patient-specific facts, numbers, or feature names from the reasoning_card.\n"
#         "- Do NOT mention the label value, and do NOT create rules like 'if feature X then label Y'.\n"
#         "- Do NOT write statistics, frequencies, or probabilities.\n"
#         "- Assume budgets are already trimmed by the caller; still output exactly two adds.\n\n"

#         "Output EXACTLY TWO JSON-Patch ops in this order:\n"
#         "1) add ONE string to /experience_memory/reasoning_templates/-\n"
#         "2) add ONE string to /experience_memory/calibration_lessons/-\n\n"

#         "JSON schema:\n"
#         "{\"patch_ops\":["
#         "{\"op\":\"add\",\"path\":\"/experience_memory/reasoning_templates/-\",\"value\":\"...\"},"
#         "{\"op\":\"add\",\"path\":\"/experience_memory/calibration_lessons/-\",\"value\":\"...\"}"
#         "]}"
#     )



# jan 3
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

    # print("__reason_cards:", len(reason_cards))
    # print("__calib_cards:", len(calib_cards))

    joined_reason = "\n\n".join(reason_cards)
    joined = "\n\n".join(calib_cards)
    mem = _memory_compact(memory)

    #jan11-sc
    return (
        "You are a physician specializing in neurodegenerative diseases who has evaluated many patients using longitudinal EHR histories "
        "and has repeatedly calibrated your clinical judgments against true ADRD outcomes. "
        "Over time, you have accumulated experience about which reasoning patterns tend to be reliable and which tend to mislead. "
        "Your task now is to maintain and refine a compact clinician experience summary that represents how an experienced doctor "
        "internally adjusts intuition after seeing many outcomes.\n\n"

        "This clinician experience is NOT a case summary and NOT a rule list. "
        "It is a distilled set of clinical instincts that guide how evidence should be weighted, how confidence should be calibrated, "
        "and how common reasoning traps should be avoided when judging whether a patient has ADRD at the current visit.\n\n"

        "You should think like a senior doctor asking:\n"
        "Across many patients, what patterns consistently deserve more trust? "
        "What patterns often looked convincing but turned out to be misleading after outcomes were known? "
        "When should I become more confident, and when should I deliberately hold back even if risk factors are present? "
        "Which alternative explanations must always be actively ruled out before concluding ADRD?\n\n"

        "Existing clinician knowledge (current version to be refined, not discarded):\n"
        f"{mem}\n\n"

        "Patient-level reasoning outputs from multiple prior evaluations:\n"
        f"{joined_reason}\n\n"

        "Corresponding calibration lessons derived from true patient outcomes:\n"
        f"{joined}\n\n"

        "Update guidance (critical):\n"
        "Integrate the new calibration lessons into the existing clinician knowledge by slightly refining, reweighting, or clarifying it. "
        "Preserve prior insights unless they are directly contradicted. "
        "If multiple lessons point in the same direction, strengthen that intuition; "
        "if lessons conflict, resolve them by favoring patterns that generalize across patients rather than rare edge cases. "
        "Avoid introducing overly specific rules; focus on tendencies, habits, and judgment calibration.\n\n"

        "Strict constraints:\n"
        "- Write exactly ONE compact paragraph.\n"
        "- Do NOT use lists, numbering, headings, or special formatting in the output.\n"
        "- Do NOT include patient-specific facts, diagnoses, medications, codes, counts, or time spans.\n"
        "- Do NOT restate individual cases or labels.\n"
        "- Express knowledge as general clinical instincts and calibration habits, not decision rules or checklists.\n"
        "- The paragraph should read like a senior clinician’s internal reasoning, not a textbook or guideline.\n\n"

        "Write the updated clinician experience paragraph now."
    )



    # return (
    #     "You are a doctor on nerodegenerative diseases, and you have reviewed patient's EHR histories from their EHR beginning to 5 years ago and new observations from 5 years ago to 1 year ago, and  conducted reasoning on whether the patient has AD or not currently. " +
    #     "You have also calibrated your reasoning against patient outcomes. " +
    #     "From this information, you now maintain and update compact clinician experience based on the reasoning process and the calibration lesson that you have learned.\n\n"

    #     "Existing clinician knowledge:\n"
    #     f"{mem}\n\n"

    #     "Patient reasoning outputs:\n"
    #     f"{joined_reason}\n\n"

    #     "Corresponding patient calibration lessons:\n"
    #     f"{joined}\n\n"

    #     # "Constraints:\n"
    #     # "- No lists, numbering, headings, or special formatting.\n"
    #     # # "- Do not include patient-specific facts, feature names, numbers, or time spans.\n"

    #     "Write the updated clinician experience paragraph now."
    # )



# jan 4
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

    # print("__reason_cards:", len(reason_cards))
    # print("__calib_cards:", len(calib_cards))

    joined_reason = "\n\n".join(reason_cards)
    joined = "\n\n".join(calib_cards)
    mem = _memory_compact(memory)

    # jan11-sc

    return (
            "You are a physician specializing in neurodegenerative diseases who has evaluated many patients using longitudinal EHR histories "
            "and has repeatedly calibrated your clinical judgments against true ADRD outcomes. "
            "Over time, you have accumulated experience about which reasoning patterns tend to be reliable and which tend to mislead. "
            "Your task now is to maintain and refine a compact clinician experience summary that represents how an experienced doctor "
            "internally adjusts intuition after seeing many outcomes.\n\n"

            "This clinician experience is NOT a case summary and NOT a rule list. "
            "It is a distilled set of clinical instincts that guide how evidence should be weighted, how confidence should be calibrated, "
            "and how common reasoning traps should be avoided when judging whether a patient has ADRD at the current visit.\n\n"

            "You should think like a senior doctor asking:\n"
            "Across many patients, what patterns consistently deserve more trust? "
            "What patterns often looked convincing but turned out to be misleading after outcomes were known? "
            "When should I become more confident, and when should I deliberately hold back even if risk factors are present? "
            "Which alternative explanations must always be actively ruled out before concluding ADRD?\n\n"

            "Existing clinician knowledge (current version to be refined, not discarded):\n"
            f"{mem}\n\n"

            "Patient-level reasoning outputs from multiple prior evaluations:\n"
            f"{joined_reason}\n\n"

            "Corresponding calibration lessons derived from true patient outcomes:\n"
            f"{joined}\n\n"

            "Update guidance (critical):\n"
            "Integrate the new calibration lessons into the existing clinician knowledge by slightly refining, reweighting, or clarifying it. "
            "Preserve prior insights unless they are directly contradicted. "
            "If multiple lessons point in the same direction, strengthen that intuition; "
            "if lessons conflict, resolve them by favoring patterns that generalize across patients rather than rare edge cases. "
            "Avoid introducing overly specific rules; focus on tendencies, habits, and judgment calibration.\n\n"

            "Strict constraints:\n"
            "- Write exactly ONE compact paragraph.\n"
            "- Do NOT use lists, numbering, headings, or special formatting in the output.\n"
            "- Do NOT include patient-specific facts, diagnoses, medications, codes, counts, or time spans.\n"
            "- Do NOT restate individual cases or labels.\n"
            "- Express knowledge as general clinical instincts and calibration habits, not decision rules or checklists.\n"
            "- The paragraph should read like a senior clinician’s internal reasoning, not a textbook or guideline.\n\n"

            "Write the updated clinician experience paragraph now."
        )



    # return (
    #     "You are a doctor on nerodegenerative diseases, and you have reviewed patient's EHR histories from their EHR beginning to 5 years ago and new observations from 5 years ago to 1 year ago, and  conducted reasoning on whether the patient has AD or not currently. " +
    #     "You have also calibrated your reasoning against patient outcomes. " +
    #     # "Now you update your existing compact clinician experience based on the reasoning process and the calibration lesson that you have learned.\n\n"
    #     "From this information, you now maintain and update compact clinician experience based on the reasoning process and the calibration lesson that you have learned.\n\n"

    #     "Existing clinician knowledge:\n"
    #     f"{mem}\n\n"

    #     "Patient reasoning outputs:\n"
    #     f"{joined_reason}\n\n"

    #     "Corresponding patient calibration lessons:\n"
    #     f"{joined}\n\n"

    #     # "Constraints:\n"
    #     # "- No lists, numbering, headings, or special formatting.\n"
    #     # # "- Do not include patient-specific facts, feature names, numbers, or time spans.\n"

    #     "Write the updated clinician experience paragraph now."
    # )



# dec 23, update version3
def _memory_compact_for_risk(memory: Dict[str, Any], max_templates: int = 6, max_lessons: int = 6) -> Dict[str, Any]:
    """
    Compact memory for risk scoring to avoid blowing up the prompt context.
    This is about prompt length stability, NOT about steering the score.
    """
    kc = memory.get("knowledge_core", {})
    em = memory.get("experience_memory", {})
    return {
        "knowledge_core": {
            "evidence_ladder": kc.get("evidence_ladder", {}),
            "rules": kc.get("rules", []),
        },
        "experience_memory": {
            "reasoning_templates": (em.get("reasoning_templates", []) or [])[:max_templates],
            "calibration_lessons": (em.get("calibration_lessons", []) or [])[:max_lessons],
        },
        "meta": memory.get("meta", {}),
    }

def risk_assessment_prompt(
    base_codes: List[str],
    delta_codes: List[str],
    meds: List[str],
    memory: Dict[str, Any],
    reasoning_card: Dict[str, Any],
) -> str:
    mem_compact = _memory_compact_for_risk(memory)

    return (
        "You are a clinician performing a 5-year Alzheimer's disease (AD) risk assessment.\n"
        "CRITICAL: Output MUST be valid JSON only.\n"
        "CRITICAL: The FIRST character of your output MUST be '{' and the LAST character MUST be '}'.\n\n"
        "Inputs:\n"
        f"base_codes: {base_codes if base_codes else []}\n"
        f"delta_codes: {delta_codes if delta_codes else []}\n"
        f"meds: {meds if meds else []}\n\n"
        "Compact memory (reference knowledge + experience):\n"
        f"{_json_dumps(mem_compact)}\n\n"
        "Reasoning card (from prior step):\n"
        f"{_json_dumps(reasoning_card)}\n\n"
        "Task:\n"
        "- Output an INTEGER risk_score_1to10 for AD within 5 years (1=very low, 10=very high).\n"
        "- Do NOT assume AD unless directly supported; keep uncertainty explicit.\n\n"
        "Reasoning requirements (format only; do NOT add extra narration):\n"
        "- risk_score_rationale: a list of reasons for the score.\n"
        "- reasoning_path: an evidence chain explaining why that score follows from the inputs.\n"
        "- reasoning_path MUST NOT contain self-instructions or generic plans "
        "(avoid 'identify risk factors', 'evaluate evidence', 'assess missing data', 'conclude').\n"
        "- Each reasoning_path item should cite concrete evidence from inputs or reasoning_card.\n\n"
        "Output STRICT JSON with exactly these keys:\n"
        "{"
        "\"risk_score_1to10\": <INTEGER 1-10>, "
        "\"risk_score_rationale\": [<STRING>, ...], "
        "\"reasoning_path\": [<STRING>, ...]"
        "}\n"
    )



# for scorer
def risk_score_prompt(
    base_codes: List[str],
    delta_codes: List[str],
    meds: List[str],
    memory: Dict[str, Any],
    reasoning_card: Dict[str, Any],
) -> str:
    """
    Phase2 second-call prompt:
    - Takes the already-generated reasoning_card + memory + EHR inputs
    - Outputs ONLY: score + short rationale + short reasoning path
    - Neutral: does not force direction; uses 5 if unclear
    """
    mem = _memory_compact(memory)

    return (
        "You are a clinician performing a cautious cognitive risk rating.\n"
        "You MUST output a single JSON object and nothing else (no prose, no markdown, no code fences).\n\n"
        "Task:\n"
        "- Rate 5-year risk of Alzheimer's disease on a 1-10 scale.\n"
        "- Be neutral and uncertainty-aware. Do NOT assume AD unless directly supported.\n"
        "- If evidence is unclear or mixed, choose 5.\n\n"
        f"base_codes: {base_codes if base_codes else []}\n"
        f"delta_codes: {delta_codes if delta_codes else []}\n"
        f"meds: {meds if meds else []}\n\n"
        "Clinician memory (reference only):\n"
        f"{_json_dumps(mem)}\n\n"
        "Reasoning card (from prior step):\n"
        f"{_json_dumps(reasoning_card)}\n\n"
        "Output STRICT JSON schema:\n"
        "{"
        "\"risk_score_1to10\": 5, "
        "\"risk_score_rationale\": [\"...\", \"...\"], "
        "\"reasoning_path\": [\"Step 1: ...\", \"Step 2: ...\", \"Step 3: ...\"]"
        "}\n"
    )


