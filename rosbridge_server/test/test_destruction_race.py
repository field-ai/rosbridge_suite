#!/usr/bin/env python3
"""
Stress tests for entity destruction races in the rosbridge executor.

These tests verify that destroying ROS entities (subscriptions, publishers)
during protocol cleanup does not crash the executor. The tests use only
the public API: IncomingQueue, RosbridgeProtocol, and executor.create_task.

The core behavioral assertion is simple: after any cleanup sequence, the
executor must still be able to process new tasks.

Because race conditions are probabilistic, these tests use high entity
counts, barrier-synchronized teardown, many iterations, and interleaved
lifecycles to maximize the probability of triggering a race if one exists.
"""

from __future__ import annotations

import json
import threading
import time
import unittest
import uuid

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rosbridge_library.rosbridge_protocol import RosbridgeProtocol
from rosbridge_server.websocket_handler import IncomingQueue


def _subscribe_msg(topic: str) -> str:
    return json.dumps({"op": "subscribe", "topic": topic, "type": "std_msgs/msg/String"})


def _advertise_msg(topic: str) -> str:
    return json.dumps({"op": "advertise", "topic": topic, "type": "std_msgs/msg/String"})


class TestDestructionRace(unittest.TestCase):
    """
    Verify that entity destruction during protocol cleanup does not crash
    the executor.

    Uses spin_once in a loop (rather than spin) so executor exceptions are
    captured per-iteration instead of killing the thread on first error.
    """

    def setUp(self) -> None:
        rclpy.init()
        self.executor = SingleThreadedExecutor()
        self.node = Node("test_destruction_race")
        self.executor.add_node(self.node)

        self.executor_errors: list[BaseException] = []
        self._stop_spinning = threading.Event()
        self.exec_thread = threading.Thread(target=self._spin_executor, daemon=True)
        self.exec_thread.start()

        self._test_id = uuid.uuid4().hex[:8]

    def _spin_executor(self) -> None:
        """Spin with per-iteration error capture — keeps running after errors."""
        while not self._stop_spinning.is_set():
            try:
                self.executor.spin_once(timeout_sec=0.05)
            except Exception as e:
                self.executor_errors.append(e)

    def tearDown(self) -> None:
        self._stop_spinning.set()
        self.exec_thread.join(timeout=10)
        self.executor.remove_node(self.node)
        self.node.destroy_node()
        self.executor.shutdown()
        rclpy.shutdown()

    def _assert_executor_healthy(self) -> None:
        """Assert the executor has not raised any errors."""
        self.assertTrue(
            self.exec_thread.is_alive(),
            f"Executor thread died unexpectedly",
        )
        self.assertEqual(
            self.executor_errors,
            [],
            f"Executor raised {len(self.executor_errors)} error(s): "
            f"{self.executor_errors[:5]}",
        )

    def _assert_executor_functional(self, msg: str = "Executor cannot process tasks") -> None:
        """Assert the executor can still process a new task."""
        done = threading.Event()
        self.executor.create_task(done.set)
        self.assertTrue(done.wait(timeout=5), msg)

    def _make_client(
        self, num_subs: int, num_pubs: int, prefix: str
    ) -> IncomingQueue:
        """Create a protocol + queue and enqueue entity creation messages."""
        protocol = RosbridgeProtocol(str(uuid.uuid4()), self.node)
        queue = IncomingQueue(protocol)
        queue.start()
        for i in range(num_subs):
            queue.push(_subscribe_msg(f"/{prefix}_s{i}"))
        for i in range(num_pubs):
            queue.push(_advertise_msg(f"/{prefix}_p{i}"))
        return queue

    # ------------------------------------------------------------------
    # Test 1: Barrier-synchronized mass disconnect
    # ------------------------------------------------------------------

    def test_barrier_synchronized_mass_disconnect(self) -> None:
        """
        Many clients disconnect at the exact same instant.

        A threading.Barrier ensures all finish() calls fire simultaneously,
        maximizing the window for entity destruction to overlap with the
        executor's wait-set rebuild.
        """
        num_clients = 30
        entities_per_client = 20

        queues: list[IncomingQueue] = []
        for i in range(num_clients):
            q = self._make_client(
                num_subs=entities_per_client,
                num_pubs=entities_per_client,
                prefix=f"barrier_{self._test_id}_c{i}",
            )
            queues.append(q)

        # Let all entities settle on the executor
        time.sleep(3.0)
        self._assert_executor_healthy()

        # Barrier ensures all finish() calls happen at the same instant
        barrier = threading.Barrier(num_clients)
        barrier_errors: list[Exception] = []

        def _finish_at_barrier(q: IncomingQueue) -> None:
            try:
                barrier.wait(timeout=10)
                q.finish()
            except Exception as e:
                barrier_errors.append(e)

        threads = [
            threading.Thread(target=_finish_at_barrier, args=(q,))
            for q in queues
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        for q in queues:
            q.join(timeout=10)

        self.assertFalse(barrier_errors, f"Barrier/finish errors: {barrier_errors}")

        # Give executor time to process all queued cleanup tasks
        time.sleep(2.0)

        self._assert_executor_healthy()

        # Verify executor is still functional with multiple probes
        for probe in range(10):
            self._assert_executor_functional(
                f"Executor dead after mass disconnect (probe {probe})"
            )

    # ------------------------------------------------------------------
    # Test 2: Repeated rapid connect/disconnect iterations
    # ------------------------------------------------------------------

    def test_repeated_rapid_connect_disconnect(self) -> None:
        """
        Run many rapid connect/disconnect cycles.

        A race condition that triggers even 5% of the time will be caught
        with >99% probability over 100 iterations. Each cycle creates
        entities and immediately tears them down without waiting, maximizing
        overlap between creation and destruction on the executor.
        """
        iterations = 100
        entities = 10

        for iteration in range(iterations):
            q = self._make_client(
                num_subs=entities,
                num_pubs=entities,
                prefix=f"rapid_{self._test_id}_{iteration}",
            )

            # Don't wait for entities to settle — finish immediately to
            # maximize overlap with the executor's wait-set construction.
            q.finish()
            q.join(timeout=5)

            self._assert_executor_functional(
                f"Executor died at iteration {iteration}"
            )

        self._assert_executor_healthy()

    # ------------------------------------------------------------------
    # Test 3: Interleaved connect and disconnect waves
    # ------------------------------------------------------------------

    def test_interleaved_connect_disconnect(self) -> None:
        """
        New clients connect while old clients disconnect simultaneously.

        This exercises the executor from both directions: new entities being
        added to the wait set while old entities are being destroyed. This
        is the most realistic traffic pattern (clients continuously arriving
        and departing).
        """
        num_waves = 20
        clients_per_wave = 10
        entities_per_client = 10

        previous_queues: list[IncomingQueue] = []

        for wave in range(num_waves):
            # Start new clients for this wave
            new_queues: list[IncomingQueue] = []
            for i in range(clients_per_wave):
                q = self._make_client(
                    num_subs=entities_per_client,
                    num_pubs=0,
                    prefix=f"wave_{self._test_id}_w{wave}_c{i}",
                )
                new_queues.append(q)

            # Simultaneously disconnect all previous-wave clients.
            # This means entity creation (new wave) and entity destruction
            # (old wave) happen concurrently on the executor.
            for q in previous_queues:
                q.finish()
            for q in previous_queues:
                q.join(timeout=5)

            previous_queues = new_queues

            self._assert_executor_functional(
                f"Executor died at wave {wave}"
            )

        # Clean up final wave
        for q in previous_queues:
            q.finish()
            q.join(timeout=5)

        time.sleep(1.0)
        self._assert_executor_healthy()

    # ------------------------------------------------------------------
    # Test 4: Combined stress — high concurrency with interleaving
    # ------------------------------------------------------------------

    def test_combined_stress(self) -> None:
        """
        Combines barrier-synchronized disconnect, rapid cycling, and
        concurrent new-client creation in a single sustained stress test.

        Three phases run back-to-back, verifying the executor survives
        sustained mixed-mode pressure.
        """
        # Phase 1: Create a large batch, barrier-disconnect, immediately
        #          start creating a new batch.
        batch_a: list[IncomingQueue] = []
        for i in range(15):
            q = self._make_client(
                num_subs=15, num_pubs=15,
                prefix=f"combo_{self._test_id}_a{i}",
            )
            batch_a.append(q)

        time.sleep(2.0)

        # Start batch B creation simultaneously with batch A destruction
        batch_b: list[IncomingQueue] = []
        barrier = threading.Barrier(len(batch_a))
        barrier_errors: list[Exception] = []

        def _finish_at_barrier(q: IncomingQueue) -> None:
            try:
                barrier.wait(timeout=10)
                q.finish()
            except Exception as e:
                barrier_errors.append(e)

        destroy_threads = [
            threading.Thread(target=_finish_at_barrier, args=(q,))
            for q in batch_a
        ]
        for t in destroy_threads:
            t.start()

        # While batch A is being destroyed, create batch B
        for i in range(15):
            q = self._make_client(
                num_subs=15, num_pubs=15,
                prefix=f"combo_{self._test_id}_b{i}",
            )
            batch_b.append(q)

        for t in destroy_threads:
            t.join(timeout=15)
        for q in batch_a:
            q.join(timeout=10)

        self.assertFalse(barrier_errors, f"Phase 1 errors: {barrier_errors}")
        self._assert_executor_functional("Executor dead after phase 1")

        # Phase 2: Rapid cycling while batch B is still alive
        for iteration in range(50):
            q = self._make_client(
                num_subs=5, num_pubs=5,
                prefix=f"combo_{self._test_id}_r{iteration}",
            )
            q.finish()
            q.join(timeout=5)

        self._assert_executor_functional("Executor dead after phase 2")

        # Phase 3: Destroy batch B
        for q in batch_b:
            q.finish()
        for q in batch_b:
            q.join(timeout=10)

        time.sleep(1.0)
        self._assert_executor_healthy()
        self._assert_executor_functional("Executor dead after phase 3")


if __name__ == "__main__":
    unittest.main()
