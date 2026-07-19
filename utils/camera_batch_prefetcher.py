import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List, Sequence


class CameraBatchPrefetcher:
    """Load exactly one scheduled camera batch ahead on a CPU worker."""

    def __init__(self, dataset):
        self._dataset = dataset
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="tide-camera-loader",
        )
        self._future: Future | None = None
        self._indices = None
        self._closed = False
        self._stats = {
            "batches": 0,
            "ready_hits": 0,
            "wait_seconds": 0.0,
        }

    def _load(self, indices):
        return [self._dataset[index] for index in indices]

    def submit(self, indices: Sequence[int]) -> None:
        if self._closed:
            raise RuntimeError("camera batch prefetcher is closed")
        key = tuple(int(index) for index in indices)
        if self._future is not None:
            if key == self._indices:
                return
            raise RuntimeError(
                "camera batch prefetch was not consumed before scheduling the next batch"
            )
        self._indices = key
        self._future = self._executor.submit(self._load, key)

    def get(self, indices: Sequence[int]) -> List[object]:
        key = tuple(int(index) for index in indices)
        if self._future is None:
            return self._load(key)
        if key != self._indices:
            raise RuntimeError(
                f"camera prefetch schedule mismatch: expected={self._indices}, got={key}"
            )

        ready = self._future.done()
        wait_start = time.perf_counter()
        cameras = self._future.result()
        self._stats["wait_seconds"] += time.perf_counter() - wait_start
        self._stats["batches"] += 1
        self._stats["ready_hits"] += int(ready)
        self._future = None
        self._indices = None
        return cameras

    def get_stats(self):
        return dict(self._stats)

    def close(self) -> None:
        if self._closed:
            return
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._closed = True
