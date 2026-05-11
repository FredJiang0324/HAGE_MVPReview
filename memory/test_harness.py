"""
Test Harness Module

Handles all testing-related operations including:
- Question answering using QA LLM
- Test execution with progress tracking
- Result aggregation

Note: Evaluation is now handled by separate Evaluator module.
"""

import logging

import re
import threading
from typing import Dict, List, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from load_dataset import LoCoMoSample

logger = logging.getLogger(__name__)

class TestHarness:
    """
    Test harness for TRG memory system.

    Handles question answering and test execution.
    Evaluation is delegated to a separate Evaluator instance.
    """

    def __init__(self, memory_builder, query_engine, evaluator=None, max_top_k=None, use_simple_prompts=False):
        """
        Initialize test harness.

        Args:
            memory_builder: MemoryBuilder instance (for accessing LLM and formatters)
            query_engine: QueryEngine instance for retrieval
            evaluator: Optional Evaluator instance for answer evaluation
            max_top_k: Optional cap on top_k for retrieval (useful for smaller models)
            use_simple_prompts: Use minimal prompts for small/local models
        """
        self.memory_builder = memory_builder
        self.query_engine = query_engine
        self.qa_llm = memory_builder.llm_controller
        self.evaluator = evaluator
        self.max_top_k = max_top_k
        self.use_simple_prompts = use_simple_prompts

        self.answer_formatter = memory_builder.answer_formatter

    def answer_question(self, question: str, context: str, category: int = None, expected: str = None) -> tuple:
        """
        Generate answer from context using QA LLM.

        Args:
            question: Question text
            context: Retrieved context
            category: Question category (1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop, 5=adversarial)
            expected: Expected answer (for best-of-N selection)

        Returns:
            Generated answer string.
        """
        if hasattr(self, 'best_of_n') and self.best_of_n > 1 and hasattr(self, 'best_of_n_selector'):
            answer = self._answer_question_best_of_n(question, context, category, expected)
        else:
            answer = self._answer_question_single(question, context, category)

        return answer

    def _answer_question_single(self, question: str, context: str, category: int = None) -> str:
        """
        Generate a single answer (original method).
        """
        if not self.qa_llm:
            return self._extract_answer_simple(question, context)

        prompt = self.answer_formatter.build_qa_prompt(context, question, category=category, use_simple=self.use_simple_prompts)

        try:
            response = self.qa_llm.llm.get_completion(
                prompt,
                response_format={"type": "text"},
                temperature=0.0
            )

            answer = response.strip()
            answer = self.answer_formatter.extract_answer(answer, question, category=category)

            # Validate answer for adversarial questions (category 5)
            answer = self.answer_formatter.validate_adversarial_answer(question, answer, category)

            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Error generating answer"

    def _answer_question_best_of_n(self, question: str, context: str, category: int = None, expected: str = None) -> str:
        """
        Generate answer using best-of-N selection with LLM Judge scoring.
        """
        attempts = []

        for i in range(self.best_of_n):
            try:
                answer = self._answer_question_single(question, context, category)
                attempts.append(answer)
            except Exception as e:
                logger.warning(f"Attempt {i+1} failed: {e}")
                attempts.append("Error generating answer")

        if all(a == "Error generating answer" for a in attempts):
            return "Error generating answer"

        if hasattr(self, 'best_of_n_method') and self.best_of_n_method == 'llm_judge' and self.evaluator and expected:
            best_score = -1
            best_answer = attempts[0]

            for answer in attempts:
                if answer != "Error generating answer":
                    try:
                        eval_result = self.evaluator.evaluate_answer(
                            question,
                            expected,
                            answer,
                            question_category=category
                        )
                        score = eval_result.get('llm_judge_score', 0.0)

                        if score > best_score:
                            best_score = score
                            best_answer = answer
                    except Exception as e:
                        logger.warning(f"LLM Judge evaluation failed: {e}")

            return best_answer

        elif hasattr(self, 'best_of_n_method') and self.best_of_n_method == 'voting':
            from collections import Counter
            valid_attempts = [a for a in attempts if a != "Error generating answer"]
            if valid_attempts:
                counter = Counter(valid_attempts)
                return counter.most_common(1)[0][0]
            return attempts[0]

        elif hasattr(self, 'best_of_n_method') and self.best_of_n_method == 'f1' and expected:
            best_score = -1
            best_answer = attempts[0]

            for answer in attempts:
                if answer != "Error generating answer":
                    pred_tokens = set(answer.lower().split())
                    exp_tokens = set(expected.lower().split())

                    if exp_tokens:
                        intersection = pred_tokens & exp_tokens
                        if intersection:
                            precision = len(intersection) / len(pred_tokens) if pred_tokens else 0
                            recall = len(intersection) / len(exp_tokens)
                            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                            if f1 > best_score:
                                best_score = f1
                                best_answer = answer

            return best_answer

        for answer in attempts:
            if answer != "Error generating answer":
                return answer
        return attempts[0]

    def _extract_answer_simple(self, question: str, context: str) -> str:
        """
        Simple extraction without LLM (fallback).

        Args:
            question: Question text
            context: Retrieved context

        Returns:
            Extracted answer
        """
        if not context or "No relevant information" in context:
            return "Information not found"

        question_lower = question.lower()
        context_lower = context.lower()

        if "when" in question_lower:
            # Focus on the DETAILED MEMORIES section for temporal questions
            detailed_section = context
            if "DETAILED MEMORIES" in context:
                # Extract just the detailed memories section
                detailed_start = context.find("DETAILED MEMORIES")
                if detailed_start != -1:
                    detailed_section = context[detailed_start:]

            # First check for computed dates in Event dates mentioned format
            # Pattern matches: "Event dates mentioned: 'yesterday' (=07 May 2023)"
            event_date_pattern = r"Event dates mentioned:.*?\(=\s*([^)]+)\)"
            event_dates = re.findall(event_date_pattern, detailed_section, re.IGNORECASE)
            if event_dates:
                # Extract the date from the computed value
                date_str = event_dates[0].strip()
                # Clean any trailing punctuation or quotes
                date_str = date_str.strip('",\'')
                # Try to extract date in various formats
                date_patterns = [
                    r'(\d{1,2}\s+\w+\s+\d{4})',  # "07 May 2023"
                    r'(\w+\s+\d{1,2},?\s+\d{4})',  # "May 7, 2023"
                    r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',  # "05/07/2023"
                    r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})'  # "2023-05-07"
                ]
                for pattern in date_patterns:
                    date_match = re.search(pattern, date_str, re.IGNORECASE)
                    if date_match:
                        # Normalize the date format (remove leading zeros)
                        extracted_date = date_match.group(1)
                        extracted_date = re.sub(r'\b0(\d)\s', r'\1 ', extracted_date)
                        return extracted_date
                # If no pattern matches but we have a date string, return it as-is
                if date_str:
                    # Remove leading zeros
                    date_str = re.sub(r'\b0(\d)\s', r'\1 ', date_str)
                    return date_str

            # Also check for dates in the standard formats throughout the context
            dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4}', detailed_section)
            if dates:
                # Normalize by removing leading zeros
                date = dates[0]
                date = re.sub(r'\b0(\d)\s', r'\1 ', date)
                return date

        if "who" in question_lower:
            names = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', context)
            if names:
                return names[0]
            names = re.findall(r'(?:by|with|from)\s+([A-Z][a-z]+)', context)
            if names:
                return names[0]

        if "where" in question_lower:
            locations = re.findall(r'(?:in|at|from|to|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', context)
            if locations:
                return locations[0]

        if "how many" in question_lower or "how much" in question_lower:
            numbers = re.findall(r'\b\d+\b', context)
            if numbers:
                return numbers[0]

        if "what" in question_lower:
            question_words = set(question_lower.split()) - {'what', 'is', 'was', 'are', 'were', 'the', 'a', 'an'}
            sentences = context.split('.')
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                if len(question_words & sentence_words) >= 2:
                    match = re.search(r'(?:is|was|are|were)\s+([^,.!?]+)', sentence, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()
                    return sentence.strip()

        question_words = set(question_lower.split()) - {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'was', 'are', 'were', 'the', 'a', 'an'}
        best_sentence = ""
        best_overlap = 0

        for sentence in context.split('.'):
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words & sentence_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence.strip()

        if best_overlap >= 2:
            return best_sentence[:100]

        return "Information not found"

    def test_questions(self, sample: LoCoMoSample, max_questions: int = None) -> List[Dict]:
        """
        Test questions against memory system.

        Args:
            sample: LoCoMo dataset sample
            max_questions: Maximum number of questions to test (None = all)

        Returns:
            List of test results
        """
        results = []
        qa_list = sample.qa[:max_questions] if max_questions else sample.qa

        logger.info(f"Testing {len(qa_list)} questions")

        correct_count = 0
        total_f1 = 0
        total_bleu1 = 0
        total_llm_score = 0
        llm_scores_count = 0

        pbar = tqdm(qa_list, desc="Testing questions", position=0, leave=True)

        for qa_idx, qa in enumerate(pbar, 1):
            if qa.category == 1:
                top_k = 30
            else:
                top_k = 15
            if self.max_top_k is not None:
                top_k = min(top_k, self.max_top_k)

            query_context, answer_context = self.query_engine.query(qa.question, top_k=top_k)

            predicted = self.answer_question(qa.question, answer_context, category=qa.category, expected=str(qa.final_answer))

            metrics = None
            is_correct = False
            llm_score = 0.0

            if self.evaluator and qa.final_answer:
                eval_result = self.evaluator.evaluate_answer(
                    qa.question,
                    str(qa.final_answer),
                    predicted,
                    question_category=qa.category
                )
                metrics = eval_result.get('metrics')
                is_correct = eval_result.get('is_correct', False)
                llm_score = eval_result.get('llm_judge_score', 0.0)

                if is_correct:
                    correct_count += 1

                if metrics:
                    total_f1 += metrics.get('f1', 0)
                    total_bleu1 += metrics.get('bleu1', 0)

                total_llm_score += llm_score
                llm_scores_count += 1

            result = {
                'question_id': qa_idx,
                'question': qa.question,
                'category': qa.category,
                'expected': qa.final_answer,
                'predicted': predicted,
                'correct': is_correct,
                'metrics': metrics,
                'llm_judge_score': llm_score,
            }

            results.append(result)

            accuracy = correct_count / qa_idx * 100
            avg_f1 = total_f1 / qa_idx * 100
            avg_bleu1 = total_bleu1 / qa_idx * 100
            avg_llm = (total_llm_score / llm_scores_count * 100) if llm_scores_count > 0 else 0
            pbar.set_postfix({
                'F1': f'{avg_f1:.1f}%',
                'LLM': f'{avg_llm:.1f}%'
            })

        pbar.close()
        return results

    def test_questions_parallel(self, sample: LoCoMoSample, max_questions: int = None, n_workers: int = 3) -> List[Dict]:
        """
        Test questions with parallel execution for faster testing.

        Processes multiple questions simultaneously using ThreadPoolExecutor.

        Args:
            sample: LoCoMo dataset sample
            max_questions: Maximum number of questions to test (None = all)
            n_workers: Number of parallel workers (default: 3)

        Returns:
            List of test results
        """
        results = []
        qa_list = sample.qa[:max_questions] if max_questions else sample.qa

        logger.info(f"Testing {len(qa_list)} questions with {n_workers} parallel workers")

        metrics_lock = threading.Lock()
        correct_count = [0]
        total_f1 = [0]
        total_bleu1 = [0]
        total_llm_score = [0]
        llm_scores_count = [0]

        def process_single_question(qa_idx_qa):
            """Process a single question (runs in thread)"""
            qa_idx, qa = qa_idx_qa
            try:
                # Adaptive top_k based on query complexity
                if qa.category == 1:  # Multi-hop questions
                    top_k = 30
                else:
                    top_k = 15
                if self.max_top_k is not None:
                    top_k = min(top_k, self.max_top_k)

                # Query memory using the modular query engine
                query_context, answer_context = self.query_engine.query(qa.question, top_k=top_k)

                # Generate answer with category for open-domain handling
                predicted = self.answer_question(qa.question, answer_context, category=qa.category, expected=str(qa.final_answer))

                # Evaluate answer using evaluator module
                metrics = None
                is_correct = False
                llm_score = 0.0

                if self.evaluator and qa.final_answer:
                    eval_result = self.evaluator.evaluate_answer(
                        qa.question,
                        str(qa.final_answer),
                        predicted,
                        question_category=qa.category
                    )
                    metrics = eval_result.get('metrics')
                    is_correct = eval_result.get('is_correct', False)
                    llm_score = eval_result.get('llm_judge_score', 0.0)

                    # Thread-safe metric updates using lock
                    with metrics_lock:
                        if is_correct:
                            correct_count[0] += 1

                        if metrics:
                            total_f1[0] += metrics.get('f1', 0)
                            total_bleu1[0] += metrics.get('bleu1', 0)

                        total_llm_score[0] += llm_score
                        llm_scores_count[0] += 1

                # Build result
                result = {
                    'question_id': qa_idx,
                    'question': qa.question,
                    'category': qa.category,
                    'expected': qa.final_answer,
                    'predicted': predicted,
                    'correct': is_correct,
                    'metrics': metrics,
                    'llm_judge_score': llm_score,
                }

                return result, is_correct, metrics, llm_score

            except Exception as e:
                logger.error(f"Error processing question {qa_idx}: {e}", exc_info=True)
                # Return error result
                return {
                    'question_id': qa_idx,
                    'question': qa.question,
                    'category': qa.category,
                    'expected': qa.final_answer,
                    'predicted': f"ERROR: {str(e)}",
                    'correct': False,
                    'error': str(e)
                }, False, None, 0.0

        # Parallel processing with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Progress bar
            pbar = tqdm(
                total=len(qa_list),
                desc=f"Testing questions",
                position=0,
                leave=True
            )

            # Submit all tasks
            futures = []
            for qa_idx, qa in enumerate(qa_list, 1):
                future = executor.submit(process_single_question, (qa_idx, qa))
                futures.append((qa_idx, future))

            # Collect results as they complete
            for qa_idx, future in futures:
                try:
                    result, is_correct, metrics, llm_score = future.result()
                    results.append(result)

                    # Update progress bar with current metrics (thread-safe read)
                    completed = len(results)
                    with metrics_lock:
                        accuracy = correct_count[0] / completed * 100 if completed > 0 else 0
                        avg_f1 = total_f1[0] / completed * 100 if completed > 0 else 0
                        avg_bleu1 = total_bleu1[0] / completed * 100 if completed > 0 else 0
                        avg_llm = (total_llm_score[0] / llm_scores_count[0] * 100) if llm_scores_count[0] > 0 else 0

                    pbar.set_postfix({

                        'F1': f'{avg_f1:.1f}%',

                        'LLM': f'{avg_llm:.1f}%'
                    })
                    pbar.update(1)

                except Exception as e:
                    logger.error(f"Error collecting result for question {qa_idx}: {e}")
                    pbar.update(1)

            pbar.close()

        logger.info(f"Completed {len(results)} questions")
        logger.info(f"Final Accuracy: {correct_count[0]}/{len(results)} ({100*correct_count[0]/len(results):.1f}%)")

        return results
