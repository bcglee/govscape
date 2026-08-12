from __future__ import annotations

import math
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .keyword import AbstractKeywordIndex
from .metadata import AbstractMetadataIndex
from .vector import AbstractVectorIndex

STRATEGY_PREFILTER = "prefilter"
STRATEGY_POSTFILTER = "postfilter"


@dataclass
class HybridSearchState:
    strategy: str
    current_k: int
    estimated_selectivity: float
    prefilter_cost: float
    postfilter_cost: float


class AbstractHybridMetadataIndex(ABC):
    def __init__(self, metadata_index: AbstractMetadataIndex):
        self.metadata_index = metadata_index

    @staticmethod
    def deduplicate_by_digest(
        distances: list[float], digests: list[str], pages: list[str]
    ) -> list[tuple[float, str, str]]:
        seen: set[str] = set()

        deduped: list[tuple[float, str, str]] = []

        for distance, digest, page in zip(distances, digests, pages, strict=False):
            if digest in seen:
                continue

            seen.add(digest)
            deduped.append((float(distance), digest, str(page)))

        return deduped

    @staticmethod
    def _apply_blacklist(rows: list[tuple[float, str, str]], blacklist: set[str]):
        return [row for row in rows if row[1] not in blacklist]

    def _metadata_size(self) -> int:
        return self.metadata_index.total_entries()

    def _estimate_selectivity(self, predicates) -> float:
        if not predicates:
            return 1.0
        return self.metadata_index.estimate_selectivity(predicates)

    def _choose_strategy(
        self, estimated_selectivity: float, target_results: int
    ) -> tuple[str, float, float]:

        # Ensure selectivity is not too close to zero.
        safe_selectivity: float = max(float(estimated_selectivity), 1e-6)

        # Cost model:
        # cost of prefiltering  = selectivity * size of metadata database
        # cost of postfiltering = k * 1/selectivity

        prefilter_cost: float = safe_selectivity * float(self._metadata_size())
        postfilter_cost: float = 10 * float(target_results) * (1.0 / safe_selectivity)

        strategy = (
            STRATEGY_PREFILTER
            if prefilter_cost <= postfilter_cost
            else STRATEGY_POSTFILTER
        )
        return strategy, prefilter_cost, postfilter_cost

    @abstractmethod
    def _index_total_entries(self) -> int:
        pass

    @abstractmethod
    def _run_prefilter(
        self,
        query_embedding,
        predicates,
        target_results,
        candidates,
    ):
        pass

    @abstractmethod
    def _run_postfilter(
        self, query_embedding, predicates, target_results, blacklist, selectivity
    ):
        pass

    def search(
        self,
        query_embedding,
        predicates,
        target_results,
        blacklist=None,
    ):
        if blacklist is None:
            blacklist = set()

        estimated_selectivity = self._estimate_selectivity(predicates)

        strategy, prefilter_cost, postfilter_cost = self._choose_strategy(
            estimated_selectivity, target_results
        )

        rows = []
        metadata = {}
        current_k = 0

        if strategy == STRATEGY_PREFILTER:
            candidates = self.metadata_index.get_candidate_digests(predicates)
            candidates = candidates.difference(blacklist)

            rows, metadata, current_k = self._run_prefilter(
                query_embedding,
                predicates,
                target_results,
                candidates,
            )

        if strategy == STRATEGY_POSTFILTER:
            rows, metadata, current_k = self._run_postfilter(
                query_embedding,
                predicates,
                target_results,
                blacklist,
                estimated_selectivity,
            )

        return (
            rows,
            metadata,
            HybridSearchState(
                strategy=strategy,
                current_k=current_k,
                estimated_selectivity=estimated_selectivity,
                prefilter_cost=prefilter_cost,
                postfilter_cost=postfilter_cost,
            ),
        )


class HybridVectorMetadataIndex(AbstractHybridMetadataIndex):
    def __init__(
        self,
        vector_index: AbstractVectorIndex,
        metadata_index: AbstractMetadataIndex,
        vector_store_key: str = "text",
    ):
        super().__init__(metadata_index=metadata_index)
        self.vector_index = vector_index
        self.vector_store_key = vector_store_key

    def _index_total_entries(self) -> int:
        return self.vector_index.total_entries()

    def _run_prefilter(self, query_embedding, predicates, target_results, candidates):
        vectors, digests, pages = self.metadata_index.get_vectors_for_digests(
            self.vector_store_key, candidates
        )
        if len(digests) == 0:
            return [], {}, 0

        query_vec = np.asarray(query_embedding, dtype=np.float32)
        if query_vec.ndim == 2:
            query_vec = query_vec[0]
        if query_vec.ndim != 1:
            raise ValueError("Query embedding must be a 1D vector or shape (1, d)")

        distances = np.linalg.norm(vectors - query_vec, axis=1)
        order = np.argsort(distances)
        ranked_distances = [float(distances[i]) for i in order]
        ranked_digests = [digests[i] for i in order]
        ranked_pages = [str(pages[i]) for i in order]
        ranked_rows = self.deduplicate_by_digest(
            ranked_distances,
            ranked_digests,
            ranked_pages,
        )

        selected_rows = ranked_rows[:target_results]
        selected_digests = [digest for _, digest, _ in selected_rows]
        metadata = self.metadata_index.search(selected_digests, predicates)
        return selected_rows, metadata, len(digests)

    def _run_postfilter(
        self,
        query_embedding,
        predicates,
        target_results,
        blacklist,
        selectivity,
    ):
        safe_selectivity = max(selectivity, 1e-6)
        current_k = int(math.ceil(target_results * (1.0 / safe_selectivity)))
        filtered_rows = []
        metadata = {}

        while len(filtered_rows) < target_results:
            distances, digests, pages = self.vector_index.search(
                query_embedding, current_k
            )
            deduped = self.deduplicate_by_digest(distances, digests, pages)
            deduped = self._apply_blacklist(deduped, blacklist)
            candidate_digests = [digest for _, digest, _ in deduped]
            metadata = self.metadata_index.search(candidate_digests, predicates)
            filtered_rows = [row for row in deduped if row[1] in metadata]

            if len(filtered_rows) >= target_results:
                break

            if current_k >= self._index_total_entries():
                break

            current_k = min(self._index_total_entries(), current_k * 2)

        return filtered_rows[:target_results], metadata, current_k


class HybridKeywordMetadataIndex(AbstractHybridMetadataIndex):
    def __init__(
        self, keyword_index: AbstractKeywordIndex, metadata_index: AbstractMetadataIndex
    ):
        super().__init__(metadata_index=metadata_index)
        self.keyword_index = keyword_index

    def _index_total_entries(self) -> int:
        return self.keyword_index.total_entries()

    def _run_prefilter(
        self,
        query_text,
        predicates,
        target_results,
        candidates,
    ):
        if not candidates:
            return [], {}, 0

        current_k = max(1, target_results)
        old_results_found = -1
        filtered_rows = []
        metadata = {}

        # Since we cannot know in advance how many results will remain after
        # deduplication, we still need a loop for keyword prefiltering.
        while len(filtered_rows) < target_results:
            distances, digests, pages = self.keyword_index.search_filtered(
                query_text,
                current_k,
                candidates,
            )
            # Keep document-level uniqueness in ranking output.
            deduped = self.deduplicate_by_digest(distances, digests, pages)
            candidate_digests = [digest for _, digest, _ in deduped]
            metadata = self.metadata_index.search(candidate_digests, predicates)
            filtered_rows = [row for row in deduped if row[1] in metadata]

            if len(filtered_rows) >= target_results:
                break
            if current_k >= self._index_total_entries():
                break

            results_found = len(filtered_rows)
            if results_found == old_results_found and not predicates:
                break
            old_results_found = results_found

            current_k = min(self._index_total_entries(), current_k * 2)

        return filtered_rows[:target_results], metadata, current_k

    def _run_postfilter(
        self, query_text, predicates, target_results, blacklist, selectivity
    ):
        safe_selectivity = max(selectivity, 1e-6)
        current_k = int(math.ceil(target_results * (1.0 / safe_selectivity)))

        filtered_rows = []
        metadata = {}

        while len(filtered_rows) < target_results:
            distances, digests, pages = self.keyword_index.search(query_text, current_k)
            deduped = self.deduplicate_by_digest(distances, digests, pages)
            deduped = self._apply_blacklist(deduped, blacklist)
            candidate_digests = [digest for _, digest, _ in deduped]
            metadata = self.metadata_index.search(candidate_digests, predicates)
            filtered_rows = [row for row in deduped if row[1] in metadata]

            if len(filtered_rows) >= target_results:
                break

            if current_k >= self._index_total_entries():
                break

            current_k = min(self._index_total_entries(), current_k * 2)

        return filtered_rows[:target_results], metadata, current_k


class HybridKeywordVectorMetadataIndex:
    def __init__(
        self,
        vector_hybrid_index: HybridVectorMetadataIndex,
        keyword_hybrid_index: HybridKeywordMetadataIndex,
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
    ):
        self.vector_hybrid_index = vector_hybrid_index
        self.keyword_hybrid_index = keyword_hybrid_index
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    @staticmethod
    def _rank_score(rank: int) -> float:
        return 1.0 / (rank + 1)

    @staticmethod
    def _merge_metadata(
        vector_metadata: dict[str, list[dict]],
        keyword_metadata: dict[str, list[dict]],
    ) -> dict[str, list[dict]]:
        merged = {**vector_metadata}
        for digest, entries in keyword_metadata.items():
            if digest in merged:
                merged[digest] = merged[digest] + entries
            else:
                merged[digest] = list(entries)
        return merged

    def _combine_rows(
        self,
        vector_rows: list[tuple[float, str, str]],
        keyword_rows: list[tuple[float, str, str]],
        target_results: int,
    ) -> list[tuple[float, str, str]]:
        combined: dict[str, dict] = {}

        for rank, row in enumerate(vector_rows):
            distance, digest, page = row
            combined[digest] = {
                "row": row,
                "vector_rank": rank,
                "keyword_rank": None,
            }

        for rank, row in enumerate(keyword_rows):
            distance, digest, page = row
            entry = combined.get(digest)
            if entry is None:
                combined[digest] = {
                    "row": row,
                    "vector_rank": None,
                    "keyword_rank": rank,
                }
            else:
                entry["keyword_rank"] = rank
                vector_rank = entry["vector_rank"]
                if vector_rank is None or self._rank_score(rank) > self._rank_score(
                    vector_rank
                ):
                    entry["row"] = row

        scored_rows: list[tuple[float, tuple[float, str, str]]] = []
        for entry in combined.values():
            score = 0.0
            if entry["vector_rank"] is not None:
                score += self.vector_weight * self._rank_score(entry["vector_rank"])
            if entry["keyword_rank"] is not None:
                score += self.keyword_weight * self._rank_score(entry["keyword_rank"])
            scored_rows.append((score, entry["row"]))

        scored_rows.sort(key=lambda item: (-item[0], item[1][0], item[1][1]))
        return [row for _, row in scored_rows[:target_results]]

    def _search_components(
        self,
        query_embedding,
        query_text,
        predicates,
        target_results,
        blacklist,
        parallel: bool = True,
    ):
        search_k_vector = min(
            self.vector_hybrid_index._index_total_entries(),
            max(target_results * 3, 20),
        )
        search_k_keyword = min(
            self.keyword_hybrid_index._index_total_entries(),
            max(target_results * 3, 20),
        )

        if parallel:
            with ThreadPoolExecutor(max_workers=2) as executor:
                vector_future = executor.submit(
                    self.vector_hybrid_index.search,
                    query_embedding,
                    predicates,
                    search_k_vector,
                    blacklist,
                )
                keyword_future = executor.submit(
                    self.keyword_hybrid_index.search,
                    query_text,
                    predicates,
                    search_k_keyword,
                    blacklist,
                )
                vector_rows, vector_metadata, vector_state = vector_future.result()
                keyword_rows, keyword_metadata, keyword_state = keyword_future.result()
        else:
            vector_rows, vector_metadata, vector_state = (
                self.vector_hybrid_index.search(
                    query_embedding,
                    predicates,
                    search_k_vector,
                    blacklist,
                )
            )
            keyword_rows, keyword_metadata, keyword_state = (
                self.keyword_hybrid_index.search(
                    query_text,
                    predicates,
                    search_k_keyword,
                    blacklist,
                )
            )

        return (
            vector_rows,
            vector_metadata,
            vector_state,
            keyword_rows,
            keyword_metadata,
            keyword_state,
        )

    def search(
        self,
        query_embedding,
        query_text,
        predicates,
        target_results,
        blacklist=None,
        parallel: bool = True,
    ):
        if blacklist is None:
            blacklist = set()

        (
            vector_rows,
            vector_metadata,
            vector_state,
            keyword_rows,
            keyword_metadata,
            keyword_state,
        ) = self._search_components(
            query_embedding,
            query_text,
            predicates,
            target_results,
            blacklist,
            parallel=parallel,
        )

        combined_rows = self._combine_rows(
            vector_rows,
            keyword_rows,
            target_results,
        )
        combined_metadata = self._merge_metadata(vector_metadata, keyword_metadata)
        combined_state = HybridSearchState(
            strategy="combined",
            current_k=max(vector_state.current_k, keyword_state.current_k),
            estimated_selectivity=max(
                vector_state.estimated_selectivity,
                keyword_state.estimated_selectivity,
            ),
            prefilter_cost=(vector_state.prefilter_cost + keyword_state.prefilter_cost),
            postfilter_cost=(
                vector_state.postfilter_cost + keyword_state.postfilter_cost
            ),
        )
        return combined_rows, combined_metadata, combined_state


class HybridTextVisualKeywordIndex:
    def __init__(
        self,
        text_vector_hybrid_index: HybridVectorMetadataIndex,
        visual_vector_hybrid_index: HybridVectorMetadataIndex,
        keyword_hybrid_index: HybridKeywordMetadataIndex,
    ):
        self.text_vector_hybrid_index = text_vector_hybrid_index
        self.visual_vector_hybrid_index = visual_vector_hybrid_index
        self.keyword_hybrid_index = keyword_hybrid_index

    @staticmethod
    def _rank_score(rank: int) -> float:
        return 1.0 / (rank + 1)

    @staticmethod
    def _merge_metadata(
        *metadatas: dict[str, list[dict]],
    ) -> dict[str, list[dict]]:
        merged: dict[str, list[dict]] = {}
        for md in metadatas:
            for digest, entries in md.items():
                if digest in merged:
                    merged[digest] = merged[digest] + entries
                else:
                    merged[digest] = list(entries)
        return merged

    def _combine_rows(
        self,
        text_rows: list[tuple[float, str, str]],
        visual_rows: list[tuple[float, str, str]],
        keyword_rows: list[tuple[float, str, str]],
        target_results: int,
        weights: dict | None = None,
    ) -> list[tuple[float, str, str]]:
        if weights is None:
            weights = {"textual": 1.0, "visual": 1.0, "keyword": 1.0}

        t_weight = float(weights.get("textual", weights.get("text", 0.0) or 0.0))
        v_weight = float(weights.get("visual", 0.0))
        k_weight = float(weights.get("keyword", 0.0))

        combined: dict[str, dict] = {}

        for rank, row in enumerate(text_rows):
            distance, digest, page = row
            combined[digest] = {
                "row": row,
                "text_rank": rank,
                "visual_rank": None,
                "keyword_rank": None,
            }

        for rank, row in enumerate(visual_rows):
            distance, digest, page = row
            entry = combined.get(digest)
            if entry is None:
                combined[digest] = {
                    "row": row,
                    "text_rank": None,
                    "visual_rank": rank,
                    "keyword_rank": None,
                }
            else:
                entry["visual_rank"] = rank

        for rank, row in enumerate(keyword_rows):
            distance, digest, page = row
            entry = combined.get(digest)
            if entry is None:
                combined[digest] = {
                    "row": row,
                    "text_rank": None,
                    "visual_rank": None,
                    "keyword_rank": rank,
                }
            else:
                entry["keyword_rank"] = rank
                # Prefer a row from higher-scoring component for tie-breaking
                text_rank = entry.get("text_rank")
                visual_rank = entry.get("visual_rank")
                if entry.get("row") is None or (
                    text_rank is None and visual_rank is None
                ):
                    entry["row"] = row

        scored_rows: list[tuple[float, tuple[float, str, str]]] = []
        for entry in combined.values():
            score = 0.0
            if entry.get("text_rank") is not None:
                score += t_weight * self._rank_score(entry["text_rank"])
            if entry.get("visual_rank") is not None:
                score += v_weight * self._rank_score(entry["visual_rank"])
            if entry.get("keyword_rank") is not None:
                score += k_weight * self._rank_score(entry["keyword_rank"])
            scored_rows.append((score, entry["row"]))

        scored_rows.sort(key=lambda item: (-item[0], item[1][0], item[1][1]))
        return [row for _, row in scored_rows[:target_results]]

    def _search_components(
        self,
        query_embedding,
        query_text,
        predicates,
        target_results,
        blacklist,
        parallel: bool = True,
    ):
        search_k_vector = min(
            self.text_vector_hybrid_index._index_total_entries(),
            max(target_results * 3, 20),
        )
        search_k_visual = min(
            self.visual_vector_hybrid_index._index_total_entries(),
            max(target_results * 3, 20),
        )
        search_k_keyword = min(
            self.keyword_hybrid_index._index_total_entries(),
            max(target_results * 3, 20),
        )

        if parallel:
            with ThreadPoolExecutor(max_workers=3) as executor:
                text_future = executor.submit(
                    self.text_vector_hybrid_index.search,
                    query_embedding,
                    predicates,
                    search_k_vector,
                    blacklist,
                )
                visual_future = executor.submit(
                    self.visual_vector_hybrid_index.search,
                    query_embedding,
                    predicates,
                    search_k_visual,
                    blacklist,
                )
                keyword_future = executor.submit(
                    self.keyword_hybrid_index.search,
                    query_text,
                    predicates,
                    search_k_keyword,
                )
                text_rows, text_metadata, text_state = text_future.result()
                visual_rows, visual_metadata, visual_state = visual_future.result()
                keyword_rows, keyword_metadata, keyword_state = keyword_future.result()
        else:
            text_rows, text_metadata, text_state = self.text_vector_hybrid_index.search(
                query_embedding, predicates, search_k_vector, blacklist
            )
            visual_rows, visual_metadata, visual_state = (
                self.visual_vector_hybrid_index.search(
                    query_embedding, predicates, search_k_visual, blacklist
                )
            )
            keyword_rows, keyword_metadata, keyword_state = (
                self.keyword_hybrid_index.search(
                    query_text, predicates, search_k_keyword
                )
            )

        return (
            text_rows,
            text_metadata,
            text_state,
            visual_rows,
            visual_metadata,
            visual_state,
            keyword_rows,
            keyword_metadata,
            keyword_state,
        )

    def search(
        self,
        query_embedding,
        query_text,
        predicates,
        target_results,
        blacklist=None,
        parallel: bool = True,
        weights: dict | None = None,
    ):
        if blacklist is None:
            blacklist = set()

        (
            text_rows,
            text_metadata,
            text_state,
            visual_rows,
            visual_metadata,
            visual_state,
            keyword_rows,
            keyword_metadata,
            keyword_state,
        ) = self._search_components(
            query_embedding,
            query_text,
            predicates,
            target_results,
            blacklist,
            parallel=parallel,
        )

        combined_rows = self._combine_rows(
            text_rows,
            visual_rows,
            keyword_rows,
            target_results,
            weights=weights,
        )
        combined_metadata = self._merge_metadata(
            text_metadata,
            visual_metadata,
            keyword_metadata,
        )
        combined_state = HybridSearchState(
            strategy="combined",
            current_k=max(
                text_state.current_k,
                visual_state.current_k,
                keyword_state.current_k,
            ),
            estimated_selectivity=max(
                text_state.estimated_selectivity,
                visual_state.estimated_selectivity,
                keyword_state.estimated_selectivity,
            ),
            prefilter_cost=(
                text_state.prefilter_cost
                + visual_state.prefilter_cost
                + keyword_state.prefilter_cost
            ),
            postfilter_cost=(
                text_state.postfilter_cost
                + visual_state.postfilter_cost
                + keyword_state.postfilter_cost
            ),
        )
        return combined_rows, combined_metadata, combined_state
