import unittest

class TestTaskManager(unittest.TestCase):
    def setUp(self):
        self.tm = TaskManager()

    def test_add_task(self):
        self.tm.add_task("task1")
        self.assertIn("task1", self.tm.tasks)
        self.assertFalse(self.tm.tasks["task1"])
        self.assertIn("task1", self.tm.dependencies)
        self.assertEqual(self.tm.dependencies["task1"], [])
        self.assertIn("task1", self.tm.task_complexities)
        self.assertIsNone(self.tm.task_complexities["task1"])

    def test_complete_task(self):
        self.tm.add_task("task1")
        self.tm.complete_task("task1")
        self.assertTrue(self.tm.tasks["task1"])

    def test_set_task_complexity(self):
        self.tm.add_task("task1")
        self.tm.set_task_complexity("task1", 85)
        self.assertEqual(self.tm.task_complexities["task1"], 85)

    def test_get_task_complexity(self):
        self.tm.add_task("task1")
        self.tm.set_task_complexity("task1", 85)
        self.assertEqual(self.tm.get_task_complexity("task1"), 85)
        self.assertEqual(self.tm.get_task_complexity("task2"), 0)

    def test_get_ready_tasks(self):
        self.tm.add_task("task1")
        self.tm.add_task("task2", dependencies=["task1"])
        self.tm.complete_task("task1")
        self.assertEqual(self.tm.get_ready_tasks(), ["task2"])

class TestCollaborativeBuildOptimizer(unittest.TestCase):
    def setUp(self):
        self.cbo = CollaborativeBuildOptimizer()

    def test_add_code_snippet(self):
        score = self.cbo.add_code_snippet("print('Hello, World!')")
        self.assertGreaterEqual(score, 50)
        self.assertLessEqual(score, 100)

    def test_add_task(self):
        self.cbo.add_task("task1")
        self.assertIn("task1", self.cbo.task_manager.tasks)
        self.assertFalse(self.cbo.task_manager.tasks["task1"])
        self.assertIn("task1", self.cbo.task_manager.dependencies)
        self.assertEqual(self.cbo.task_manager.dependencies["task1"], [])
        self.assertIn("task1", self.cbo.task_manager.task_complexities)
        self.assertIsNone(self.cbo.task_manager.task_complexities["task1"])

    def test_complete_task(self):
        self.cbo.add_task("task1")
        self.cbo.complete_task("task1")
        self.assertTrue(self.cbo.task_manager.tasks["task1"])

    def test_set_task_complexity(self):
        self.cbo.add_task("task1")
        self.cbo.set_task_complexity("task1", 85)
        self.assertEqual(self.cbo.task_manager.task_complexities["task1"], 85)

    def test_get_task_complexity(self):
        self.cbo.add_task("task1")
        self.cbo.set_task_complexity("task1", 85)
        self.assertEqual(self.cbo.get_task_complexity("task1"), 85)
        self.assertEqual(self.cbo.get_task_complexity("task2"), 0)

    def test_get_ready_tasks(self):
        self.cbo.add_task("task1")
        self.cbo.add_task("task2", dependencies=["task1"])
        self.cbo.complete_task("task1")
        self.assertEqual(self.cbo.get_ready_tasks(), ["task2"])

    def test_estimate_build_time(self):
        self.cbo.add_task("task1")
        self.cbo.set_task_complexity("task1", 85)
        estimated_time = self.cbo.estimate_build_time(1, 85)
        self.assertGreaterEqual(estimated_time, 8.5)
        self.assertLessEqual(estimated_time, 8.5)

    def test_optimize_build_process(self):
        self.cbo.add_task("task1")
        self.cbo.add_task("task2", dependencies=["task1"])
        self.cbo.add_task("task3", dependencies=["task1"])
        self.cbo.add_task("task4", dependencies=["task2", "task3"])
        self.cbo.set_task_complexity("task1", 85)
        self.cbo.set_task_complexity("task2", 90)
        self.cbo.set_task_complexity("task3", 80)
        self.cbo.set_task_complexity("task4", 95)
        self.cbo.optimize_build_process()

class TestBudgetBuddyFunctional(unittest.TestCase):
    def setUp(self):
        self.cbo = CollaborativeBuildOptimizer()

    def test_full_build_process(self):
        self.cbo.add_task("task1")
        self.cbo.add_task("task2", dependencies=["task1"])
        self.cbo.add_task("task3", dependencies=["task1"])
        self.cbo.add_task("task4", dependencies=["task2", "task3"])
        self.cbo.set_task_complexity("task1", 85)
        self.cbo.set_task_complexity("task2", 90)
        self.cbo.set_task_complexity("task3", 80)
        self.cbo.set_task_complexity("task4", 95)
        self.cbo.optimize_build_process()

if __name__ == "__main__":
    unittest.main()
