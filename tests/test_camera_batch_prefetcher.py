import threading
import unittest

from utils.camera_batch_prefetcher import CameraBatchPrefetcher


class _Dataset:
    def __init__(self):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()

    def __getitem__(self, index):
        self.started.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("camera loader test was not released")
        self.calls.append(index)
        return f"camera-{index}"


class CameraBatchPrefetcherTest(unittest.TestCase):
    def test_loads_ahead_and_preserves_schedule_order(self):
        dataset = _Dataset()
        prefetcher = CameraBatchPrefetcher(dataset)
        self.addCleanup(prefetcher.close)

        prefetcher.submit([3, 1, 2])
        self.assertTrue(dataset.started.wait(timeout=1.0))
        dataset.release.set()
        self.assertEqual(
            prefetcher.get([3, 1, 2]),
            ["camera-3", "camera-1", "camera-2"],
        )
        self.assertEqual(dataset.calls, [3, 1, 2])

    def test_rejects_schedule_mismatch(self):
        dataset = _Dataset()
        dataset.release.set()
        prefetcher = CameraBatchPrefetcher(dataset)
        self.addCleanup(prefetcher.close)

        prefetcher.submit([0, 1])
        with self.assertRaisesRegex(RuntimeError, "schedule mismatch"):
            prefetcher.get([1, 0])


if __name__ == "__main__":
    unittest.main()
