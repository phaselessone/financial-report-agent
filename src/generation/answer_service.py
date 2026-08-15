"""P1 module extracted from src/generation/answerer.py (behavior preserved)."""
from __future__ import annotations

from pathlib import Path
from src.generation.abstain import compute_confidence_label, decide_abstention
from src.generation.citation_builder import build_citations
from src.generation.evidence_selector import _finalize_used_evidence_ids, _normalize_theme_candidates, _prepare_evidence, _refine_citation_evidence_ids, _selected_domain_mismatch, summarize_doc_candidates, summarize_retrieval_scores, summarize_selected_evidence
from src.generation.payload_parser import parse_model_json
from src.generation.payload_repair import _fallback_payload, _repair_answer_payload
from src.generation.prompt import SYSTEM_PROMPT, build_answer_prompt, build_comparison_focus_prompt, build_comparison_synthesis_prompt, build_inductive_observation_prompt, build_inductive_synthesis_prompt
from src.generation.routing import FALLBACK_ANSWER, _fallback_abstain_reason, infer_answer_mode, infer_fact_subtype, infer_question_type, is_numeric_or_table_query, resolve_query_domain_buckets
from src.generation.support_validator import validate_answer_support
from src.retrieval.model_store import ensure_model_downloaded
from src.utils.text_utils import normalize_text
from time import perf_counter
from typing import Any
import torch


class LocalEvidenceAnswerer:
    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: Path,
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.8,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = ensure_model_downloaded(model_name, cache_dir)
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.llm_provider = "local"
        self.llm_model = model_name
        self.model_name = model_name
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            dtype=torch_dtype if self.device.type == "cuda" else None,
        )
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _generate_json_payload(self, prompt_builder: Any, **prompt_kwargs: Any) -> tuple[dict[str, Any] | None, str, int]:
        raw_model_output = ""
        json_parse_failures = 0
        parsed_payload: dict[str, Any] | None = None
        for strict_json in (False, True):
            prompt = prompt_builder(strict_json=strict_json, **prompt_kwargs)
            raw_model_output = self._generate(prompt)
            parsed_payload = parse_model_json(raw_model_output)
            if parsed_payload is not None:
                break
            json_parse_failures += 1
        return parsed_payload, raw_model_output, json_parse_failures

    def _generate_comparison_payload(
        self,
        *,
        query: str,
        selected_evidence: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, str, int]:
        focus_payload, focus_output, focus_failures = self._generate_json_payload(
            build_comparison_focus_prompt,
            query=query,
            evidence_rows=selected_evidence,
        )
        if focus_payload is None:
            return None, focus_output, focus_failures
        doc_focus_map = {
            str(doc_id): normalize_text(str(summary))
            for doc_id, summary in dict(focus_payload.get("doc_focus_map", {}) or {}).items()
            if normalize_text(str(summary))
        }
        synthesis_payload, synthesis_output, synthesis_failures = self._generate_json_payload(
            build_comparison_synthesis_prompt,
            query=query,
            evidence_rows=selected_evidence,
            doc_focus_map=doc_focus_map,
        )
        if synthesis_payload is None:
            return None, f"{focus_output}\n\n---\n\n{synthesis_output}", focus_failures + synthesis_failures
        synthesis_payload["doc_focus_map"] = doc_focus_map
        merged_used_evidence_ids = list(
            dict.fromkeys(
                list(focus_payload.get("used_evidence_ids", []))
                + list(synthesis_payload.get("used_evidence_ids", []))
            )
        )
        synthesis_payload["used_evidence_ids"] = merged_used_evidence_ids
        return synthesis_payload, f"{focus_output}\n\n---\n\n{synthesis_output}", focus_failures + synthesis_failures

    def _generate_inductive_payload(
        self,
        *,
        query: str,
        selected_evidence: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, str, int]:
        observation_payload, observation_output, observation_failures = self._generate_json_payload(
            build_inductive_observation_prompt,
            query=query,
            evidence_rows=selected_evidence,
        )
        if observation_payload is None:
            return None, observation_output, observation_failures
        per_doc_observation = {
            str(doc_id): normalize_text(str(summary))
            for doc_id, summary in dict(observation_payload.get("per_doc_observation", {}) or {}).items()
            if normalize_text(str(summary))
        }
        theme_candidates = _normalize_theme_candidates(observation_payload.get("theme_candidates", {}) or {})
        synthesis_payload, synthesis_output, synthesis_failures = self._generate_json_payload(
            build_inductive_synthesis_prompt,
            query=query,
            evidence_rows=selected_evidence,
            per_doc_observation=per_doc_observation,
            theme_candidates=theme_candidates,
        )
        if synthesis_payload is None:
            return None, f"{observation_output}\n\n---\n\n{synthesis_output}", observation_failures + synthesis_failures
        synthesis_payload["per_doc_observation"] = per_doc_observation
        synthesis_payload["theme_candidates"] = theme_candidates
        merged_used_evidence_ids = list(
            dict.fromkeys(
                list(observation_payload.get("used_evidence_ids", []))
                + list(synthesis_payload.get("used_evidence_ids", []))
            )
        )
        synthesis_payload["used_evidence_ids"] = merged_used_evidence_ids
        return synthesis_payload, f"{observation_output}\n\n---\n\n{synthesis_output}", observation_failures + synthesis_failures

    def answer(
        self,
        *,
        query: str,
        question_type: str,
        retrieval_result: dict[str, Any],
        query_domain_hint: str = "",
    ) -> dict[str, Any]:
        question_type = question_type or infer_question_type(query)
        answer_mode = infer_answer_mode(query, question_type)
        fact_subtype = infer_fact_subtype(query, answer_mode)
        query_domain_buckets = resolve_query_domain_buckets(query, domain_hint=query_domain_hint)
        query_domain_bucket = query_domain_buckets[0] if query_domain_buckets else ""
        numeric_query = is_numeric_or_table_query(query)
        selected_evidence, doc_candidates, selected_doc_ids, forced_abstain_reason, doc_guard_triggered = _prepare_evidence(
            query=query,
            question_type=question_type,
            answer_mode=answer_mode,
            fact_subtype=fact_subtype,
            query_domain_bucket=query_domain_bucket,
            query_domain_buckets=query_domain_buckets,
            retrieval_result=retrieval_result,
        )
        retrieval_scores = summarize_retrieval_scores(retrieval_result, selected_evidence)
        doc_candidate_summary = summarize_doc_candidates(doc_candidates)
        selected_domain_buckets = sorted({candidate.get("domain_bucket", "") for candidate in doc_candidates if candidate["doc_id"] in set(selected_doc_ids)})
        citation_rule_applied = "doc_level_report_lookup" if answer_mode == "report_lookup" else "default"

        if forced_abstain_reason is not None:
            return self._abstain_payload(
                query=query,
                question_type=question_type,
                answer_mode=answer_mode,
                fact_subtype=fact_subtype,
                selected_evidence=selected_evidence,
                retrieval_scores=retrieval_scores,
                doc_candidates=doc_candidate_summary,
                selected_doc_ids=selected_doc_ids,
                query_domain_bucket=query_domain_bucket,
                selected_domain_buckets=selected_domain_buckets,
                doc_guard_triggered=doc_guard_triggered,
                citation_rule_applied=citation_rule_applied,
                abstain_reason=forced_abstain_reason,
                generation_latency_ms=0.0,
                json_parse_failures=0,
            )

        generation_start = perf_counter()
        raw_model_output = ""
        json_parse_failures = 0
        parsed_payload: dict[str, Any] | None = None
        fallback_used = False
        fallback_source: str | None = None
        title_resolved_from = "short_title_from_file_name" if answer_mode == "report_lookup" else ""

        if answer_mode == "comparison":
            parsed_payload, raw_model_output, json_parse_failures = self._generate_comparison_payload(
                query=query,
                selected_evidence=selected_evidence,
            )
        elif answer_mode == "inductive":
            parsed_payload, raw_model_output, json_parse_failures = self._generate_inductive_payload(
                query=query,
                selected_evidence=selected_evidence,
            )
        else:
            parsed_payload, raw_model_output, json_parse_failures = self._generate_json_payload(
                build_answer_prompt,
                query=query,
                question_type=question_type,
                answer_mode=answer_mode,
                evidence_rows=selected_evidence,
                numeric_query=numeric_query,
            )
        if parsed_payload is None:
            parsed_payload = _fallback_payload(answer_mode, selected_evidence, fact_subtype=fact_subtype, query=query)
            fallback_used = True
            fallback_source = "invalid_model_json"

        generation_latency_ms = round((perf_counter() - generation_start) * 1000, 2)
        if fallback_source == "invalid_model_json":
            payload = parsed_payload or {}
        else:
            payload, repair_fallback_reason = _repair_answer_payload(
                query=query,
                answer_mode=answer_mode,
                fact_subtype=fact_subtype,
                payload=parsed_payload or {},
                selected_evidence=selected_evidence,
            )
            if repair_fallback_reason is not None:
                fallback_used = True
                fallback_source = repair_fallback_reason
        used_evidence_ids = _finalize_used_evidence_ids(
            parsed_ids=payload.get("used_evidence_ids", []),
            selected_evidence=selected_evidence,
            question_type=question_type,
            answer_mode=answer_mode,
        )
        final_answer = payload.get("final_answer", FALLBACK_ANSWER)
        evidence_summary = payload.get("evidence_summary", "")
        uncertainty_note = payload.get("uncertainty_note", "")

        support_validation = validate_answer_support(
            query=query,
            question_type=question_type,
            final_answer=final_answer,
            evidence_summary=evidence_summary,
            used_evidence_ids=used_evidence_ids,
            evidence_rows=selected_evidence,
            answer_mode=answer_mode,
        )
        support_validation["domain_mismatch"] = _selected_domain_mismatch(
            query_domain_buckets,
            selected_domain_buckets,
            answer_mode=answer_mode,
        )

        if answer_mode != "report_lookup" and not support_validation.get("supported", False):
            fallback_payload = _fallback_payload(answer_mode, selected_evidence, fact_subtype=fact_subtype, query=query)
            fallback_used = True
            fallback_source = "unsupported_answer"
            used_evidence_ids = _finalize_used_evidence_ids(
                parsed_ids=fallback_payload.get("used_evidence_ids", []),
                selected_evidence=selected_evidence,
                question_type=question_type,
                answer_mode=answer_mode,
            )
            final_answer = fallback_payload["final_answer"]
            evidence_summary = fallback_payload["evidence_summary"]
            uncertainty_note = fallback_payload["uncertainty_note"]
            support_validation = validate_answer_support(
                query=query,
                question_type=question_type,
                final_answer=final_answer,
                evidence_summary=evidence_summary,
                used_evidence_ids=used_evidence_ids,
                evidence_rows=selected_evidence,
                answer_mode=answer_mode,
            )
            support_validation["domain_mismatch"] = _selected_domain_mismatch(
                query_domain_buckets,
                selected_domain_buckets,
                answer_mode=answer_mode,
            )

        effective_json_failures = json_parse_failures
        fallback_reason = _fallback_abstain_reason(final_answer=final_answer, fallback_source=fallback_source)
        referenced_rows = [row for row in selected_evidence if row["evidence_id"] in set(used_evidence_ids)] or selected_evidence
        abstained, abstain_reason = decide_abstention(
            question_type=question_type,
            evidence_rows=referenced_rows,
            retrieval_scores=retrieval_scores,
            support_validation=support_validation,
            json_parse_failures=effective_json_failures,
            answer_mode=answer_mode,
            fallback_reason=fallback_reason,
        )
        confidence_label = compute_confidence_label(
            retrieval_scores=retrieval_scores,
            support_validation=support_validation,
            abstained=abstained,
        )
        citation_evidence_ids = []
        if not abstained:
            citation_payload = dict(payload)
            citation_payload.update(
                {
                    "final_answer": final_answer,
                    "evidence_summary": evidence_summary,
                    "uncertainty_note": uncertainty_note,
                }
            )
            citation_evidence_ids = _refine_citation_evidence_ids(
                answer_mode=answer_mode,
                payload=citation_payload,
                selected_evidence=selected_evidence,
                used_evidence_ids=used_evidence_ids,
            )
        citations = build_citations(used_evidence_ids=citation_evidence_ids, evidence_rows=selected_evidence) if not abstained else []
        used_evidence_ids = citation_evidence_ids if not abstained else []
        return {
            "query": query,
            "question_type": question_type,
            "query_intent": answer_mode,
            "answer_mode": answer_mode,
            "fact_subtype": fact_subtype,
            "llm_provider": getattr(self, "llm_provider", "local"),
            "llm_model": getattr(self, "llm_model", getattr(self, "model_name", "")),
            "answer_source": "deterministic" if fallback_used else "llm",
            "fallback_used": fallback_used,
            "fallback_reason": fallback_source,
            "final_answer": FALLBACK_ANSWER if abstained else final_answer,
            "evidence_summary": "" if abstained else evidence_summary,
            "uncertainty_note": uncertainty_note,
            "used_evidence_ids": [] if abstained else used_evidence_ids,
            "abstained": abstained,
            "abstain_reason": abstain_reason,
            "abstain_gate": abstain_reason or "none",
            "confidence_label": confidence_label,
            "citations": citations,
            "retrieval_scores": retrieval_scores,
            "support_validation": support_validation,
            "matched_numeric_tokens": support_validation.get("matched_numeric_tokens", []),
            "title_resolved_from": title_resolved_from,
            "query_domain_bucket": query_domain_bucket,
            "query_domain_buckets": query_domain_buckets,
            "selected_domain_buckets": selected_domain_buckets,
            "doc_guard_triggered": doc_guard_triggered,
            "citation_rule_applied": citation_rule_applied,
            "support_filter_applied": support_validation.get("support_filter_applied", "default"),
            "selected_evidence": summarize_selected_evidence(selected_evidence),
            "doc_candidates": doc_candidate_summary,
            "selected_doc_ids": selected_doc_ids,
            "raw_model_output": raw_model_output,
            "json_parse_failures": json_parse_failures,
            "generation_latency_ms": generation_latency_ms,
        }

    def _generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            rendered = f"{SYSTEM_PROMPT}\n\n{prompt}"

        inputs = self.tokenizer(rendered, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        prompt_length = inputs["input_ids"].shape[-1]
        output_ids = generated[0][prompt_length:]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    def _abstain_payload(
        self,
        *,
        query: str,
        question_type: str,
        answer_mode: str,
        fact_subtype: str,
        selected_evidence: list[dict[str, Any]],
        retrieval_scores: dict[str, float],
        doc_candidates: list[dict[str, Any]],
        selected_doc_ids: list[str],
        query_domain_bucket: str,
        selected_domain_buckets: list[str],
        doc_guard_triggered: bool,
        citation_rule_applied: str,
        abstain_reason: str,
        generation_latency_ms: float,
        raw_model_output: str = "",
        json_parse_failures: int = 0,
    ) -> dict[str, Any]:
        support_validation = validate_answer_support(
            query=query,
            question_type=question_type,
            final_answer=FALLBACK_ANSWER,
            evidence_summary="",
            used_evidence_ids=[],
            evidence_rows=selected_evidence,
            answer_mode=answer_mode,
        )
        support_validation["domain_mismatch"] = False
        return {
            "query": query,
            "question_type": question_type,
            "query_intent": answer_mode,
            "answer_mode": answer_mode,
            "fact_subtype": fact_subtype,
            "llm_provider": getattr(self, "llm_provider", "local"),
            "llm_model": getattr(self, "llm_model", getattr(self, "model_name", "")),
            "answer_source": "deterministic",
            "fallback_used": True,
            "fallback_reason": abstain_reason,
            "final_answer": FALLBACK_ANSWER,
            "evidence_summary": "",
            "uncertainty_note": "基于当前证据进行保守回答。",
            "used_evidence_ids": [],
            "abstained": True,
            "abstain_reason": abstain_reason,
            "abstain_gate": abstain_reason,
            "confidence_label": "low",
            "citations": [],
            "retrieval_scores": retrieval_scores,
            "support_validation": support_validation,
            "matched_numeric_tokens": support_validation.get("matched_numeric_tokens", []),
            "title_resolved_from": "short_title_from_file_name" if answer_mode == "report_lookup" else "",
            "query_domain_bucket": query_domain_bucket,
            "selected_domain_buckets": selected_domain_buckets,
            "doc_guard_triggered": doc_guard_triggered,
            "citation_rule_applied": citation_rule_applied,
            "support_filter_applied": support_validation.get("support_filter_applied", "default"),
            "selected_evidence": summarize_selected_evidence(selected_evidence),
            "doc_candidates": doc_candidates,
            "selected_doc_ids": selected_doc_ids,
            "raw_model_output": raw_model_output,
            "json_parse_failures": json_parse_failures,
            "generation_latency_ms": generation_latency_ms,
        }
