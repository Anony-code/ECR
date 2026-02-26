# memory_schema.py
from __future__ import annotations

# def init_memory():
#     """
#     Static knowledge_core + evolving experience_memory.
#     Keep this file minimal: DO NOT put pipeline/training code here.
#     """
#     return {
#         "knowledge_core": {
#             "evidence_ladder": {
#                 "direct_neurodegeneration": [
#                     "documented cognitive impairment / memory loss",
#                     "MCI/dementia diagnosis code",
#                     "AD diagnosis code",
#                     "positive amyloid/tau biomarkers",
#                 ],
#                 "vascular_pathway": [
#                     "atrial fibrillation/arrhythmia",
#                     "stroke/TIA",
#                     "hypertension",
#                     "diabetes",
#                     "CKD",
#                     "hyperlipidemia",
#                 ],
#                 "reversible_causes": [
#                     "depression",
#                     "thyroid disease",
#                     "vitamin B12 deficiency",
#                     "sleep apnea",
#                     "medication side effects (anticholinergics/benzos/opioids)",
#                 ],
#             },
#             "rules": [
#                 "Do not assume Alzheimer's disease unless directly supported.",
#                 "Always separate supported evidence from inferred mechanisms.",
#                 "State uncertainty explicitly.",
#             ],
#         },
#         "experience_memory": {
#             "reasoning_templates": [],
#             "calibration_lessons": [],
#         },
#         "meta": {"version": 1},
#     }



# Jan 3, update with only one paragraph for experience memory
def init_memory():
    """
    Static knowledge_core + evolving experience_memory.
    Keep this file minimal: DO NOT put pipeline/training code here.
    """
    return {
        "knowledge_core": {
            "evidence_ladder": {
                "direct_neurodegeneration": [
                    "documented cognitive impairment / memory loss",
                    "MCI/dementia diagnosis code",
                    "AD diagnosis code",
                    "positive amyloid/tau biomarkers",
                ],
                "vascular_pathway": [
                    "atrial fibrillation/arrhythmia",
                    "stroke/TIA",
                    "hypertension",
                    "diabetes",
                    "CKD",
                    "hyperlipidemia",
                ],
                "reversible_causes": [
                    "depression",
                    "thyroid disease",
                    "vitamin B12 deficiency",
                    "sleep apnea",
                    "medication side effects (anticholinergics/benzos/opioids)",
                ],
            },
            "rules": [
                "Do not assume Alzheimer's disease unless directly supported.",
                # "Always separate supported evidence from inferred mechanisms.",
                # "State uncertainty explicitly.",
            ],
        },
        "experience_knowledge": "",
        "meta": {"version": 1},
    }



# def init_memory_pd():
#     """
#     Minimal PD knowledge_core for EHR-based baseline interpretation.
#     """
#     return {
#         "knowledge_core": {
#             "evidence_ladder": {
#                 "motor_anchors": [
#                     "anchor parkinsonism on bradykinesia; rest tremor/rigidity are supporting features",
#                     "gait/falls are nonspecific without motor anchors",
#                 ],
#                 "non_motor_context": [
#                     "RBD-like sleep and autonomic burden (constipation/orthostasis/urinary) are potential prodromal context",
#                     "hyposmia is supportive context; mood symptoms are low-specificity and mainly confounding unless paired with stronger patterns",
#                 ],
#                 "ehr_proxies_and_medications": [
#                     "tremor/gait/falls and mobility-support signals (PT/OT, aids) reflect functional motor burden",
#                     "dopaminergic exposure is supportive context, not definitive evidence",
#                 ],
#             },
#             "rules": [
#                 "Do not assume Parkinson's disease unless directly supported.",

#                 # "Prefer bradykinesia-centered patterns and longitudinal coherence over single symptoms or codes.",
#                 # "Treat medications as context, not proof; state uncertainty when anchors are weak or documentation is sparse.",
#             ],
#         },
#         "experience_knowledge": "",
#         "meta": {"version": 1, "condition": "pd"},
#     }

def init_memory_pd():
    """
    PD init memory (symptom/concept list; minimal priors).
    """
    return {
        "knowledge_core": {
            "evidence_ladder": {
                "motor_features": [
                    "bradykinesia",
                    "rigidity",
                    "rest tremor",
                    "postural instability",
                    "gait disturbance",
                    "falls",
                    "hypophonia",
                    "micrographia",
                    "masked facies",
                ],
                "non_motor_features": [
                    "REM sleep behavior disorder (RBD)",
                    "hyposmia / loss of smell",
                    "constipation",
                    "orthostatic hypotension / lightheadedness",
                    "urinary urgency/frequency/incontinence",
                    "depression/anxiety/apathy",
                    "cognitive impairment / executive dysfunction",
                    "visual hallucinations",
                    "fatigue",
                ],
                "medication_and_related_signals": [
                    "dopaminergic therapy (levodopa, dopamine agonists)",
                    "MAO-B inhibitors / COMT inhibitors",
                    "dopamine-blocking agents (antipsychotics, metoclopramide, prochlorperazine)",
                    "anticholinergic/sedative burden (benzos, Z-drugs, antihistamines, TCAs, opioids)",
                ],
                "common_mimics": [
                    "stroke/TIA and cerebrovascular disease",
                    "degenerative spine or joint disease",
                    "essential tremor",
                    "drug-induced parkinsonism",
                    "delirium/acute encephalopathy",
                ],
            },
         "rules": [
                "Do not assume Parkinson's disease unless directly supported.",
                "Absence of medications may or may not reflect true absence but incomplete documentation.",
                ]
},
        "experience_knowledge": "",
        "meta": {"version":1},
    }


def init_memory_adrd():
    """
    ADRD early-prediction init memory (~5y pre-onset), mechanism-aware, sparse-EHR friendly.
    Minimal + precise: keep only high-signal early features, modifiers, and key mimics.
    """
    return {
        "knowledge_core": {
            "evidence_ladder": {
                # Cross-subtype prodromal syndrome (earliest structured-EHR signals)
                "prodromal_cognitive_functional": [
                    "subjective cognitive decline / memory complaint",
                    "mild cognitive impairment (MCI) / mild cognitive disorder",
                    "executive dysfunction / attention impairment",
                    "new functional difficulty (IADL/ADL decline)",
                    "neuropsychiatric change accompanying cognitive concern (depression, anxiety, apathy)",
                ],

                # AD-tilting signals (amyloid/tau pathway; non-diagnostic, use as tilt)
                "alzheimer's_disease_tilt": [
                    "amnestic MCI / memory-dominant impairment",
                    "late-life mood symptoms in cognitive context (depression/anxiety)",
                    "sleep disturbance in cognitive context (insomnia/circadian disruption)",
                ],

                # Vascular contribution (often mixed; require direct cerebrovascular link or heavy burden as modifier)
                "vascular_dementia_tilt": [
                    "stroke/TIA history with cognitive concern",
                    "atrial fibrillation / major arrhythmia",
                    "high vascular burden (hypertension + diabetes/hyperlipidemia/CKD)",
                    "heart failure / chronic hypoperfusion states",
                ],

                # DLB-tilting signals (alpha-synuclein; prodrome often sleep/autonomic/neuropsychiatric)
                "dementia_with_lewy_bodies_tilt": [
                    "REM sleep behavior disorder",
                    "recurrent well-formed visual hallucinations (non-delirium)",
                    "parkinsonism (bradykinesia/rigidity/gait instability) before dementia code",
                    "autonomic dysfunction in neuro context (orthostatic hypotension, constipation, urinary dysfunction)",
                ],

                # FTD-tilting signals (FTLD; commonly mis-coded as psychiatric without cognitive framing)
                "frontotemporal_dementia_tilt": [
                    "new behavioral disinhibition/apathy/compulsions with cognitive concern",
                    "new language impairment/aphasia / primary progressive aphasia signals",
                ],

                # Susceptibility / reserve modifiers (risk modifiers, not etiologic evidence)
                "susceptibility_modifiers": [
                    "older age",
                    "hearing loss",
                    "social isolation proxies (if available)",
                ],

                # Mimics / reversible contributors (separate from prodrome in sparse EHR)
                "acute_or_reversible_modifiers": [
                    "delirium / acute encephalopathy",
                    "thyroid dysfunction",
                    "vitamin B12 deficiency / severe anemia",
                    "obstructive sleep apnea",
                    "high anticholinergic / sedative burden (anticholinergics, benzos, Z-drugs, opioids)",
                    "substance-related cognitive symptoms",
                ],
            },
            # "rules": [
            #     "Treat ADRD as an umbrella concept; focus on whether evidence supports a coherent cognitive and functional decline pattern rather than subtype labeling.",
            #     # "Prefer bradykinesia-centered patterns and longitudinal coherence over single symptoms or codes.",
            #     # "Treat medications as context, not proof; state uncertainty when anchors are weak or documentation is sparse.",
            # ],
            "rules": [
                "Treat ADRD as an umbrella concept; focus on whether evidence supports a coherent cognitive and functional decline pattern rather than subtype labeling.",
                "If baseline evidence does not support a coherent cognitive or functional pattern, it is acceptable to not forcing prodromal framing.",
            ],
        },
        "experience_knowledge": "",
        "meta": {"version": 1},
    }
