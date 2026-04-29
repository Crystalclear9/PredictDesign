import random

class MealMaster:
    """
    A program to personalize meal plans based on dietary preferences, health goals, and available ingredients.
    """
    def __init__(self):
        self.recipes = {
            "Grilled Chicken Salad": {"calories": 350, "protein": 30, "carbs": 20, "fat": 15, "ingredients": ["chicken", "lettuce", "tomato", "olive oil"]},
            "Vegetable Stir Fry": {"calories": 400, "protein": 20, "carbs": 30, "fat": 20, "ingredients": ["broccoli", "carrot", "soy sauce", "rice"]},
            "Peanut Butter Banana Sandwich": {"calories": 500, "protein": 10, "carbs": 50, "fat": 20, "ingredients": ["peanut butter", "banana", "bread"]}
        }
        self.user_preferences = {}

    def gather_user_preferences(self):
        """
        Gathers user input regarding dietary preferences, health goals, and available ingredients.
        """
        self.user_preferences['calories'] = int(input("Enter your daily calorie limit: "))
        self.user_preferences['protein'] = int(input("Enter your daily protein goal (in grams): "))
        self.user_preferences['carbs'] = int(input("Enter your daily carb goal (in grams): "))
        self.user_preferences['fat'] = int(input("Enter your daily fat goal (in grams): "))
        self.user_preferences['allergies'] = input("Enter any food allergies (comma-separated): ").split(', ')
        self.user_preferences['ingredients'] = input("Enter available ingredients (comma-separated): ").split(', ')

    def generate_meal_plan(self):
        """
        Generates a personalized meal plan based on user preferences.
        """
        meal_plan = []
        for recipe, details in self.recipes.items():
            if all(ingredient in self.user_preferences['ingredients'] for ingredient in details['ingredients']) and \
               recipe not in self.user_preferences['allergies'] and \
               details['calories'] <= self.user_preferences['calories'] and \
               details['protein'] <= self.user_preferences['protein'] and \
               details['carbs'] <= self.user_preferences['carbs'] and \
               details['fat'] <= self.user_preferences['fat']:
                meal_plan.append(recipe)
        return meal_plan

    def display_meal_plan(self, meal_plan):
        """
        Displays the generated meal plan.
        :param meal_plan: A list of recommended recipes.
        """
        if meal_plan:
            print("Your personalized meal plan:")
            for meal in meal_plan:
                print(f"- {meal}")
        else:
            print("No meal plans match your preferences. Please adjust your preferences and try again.")

# Example usage
if __name__ == "__main__":
    meal_master = MealMaster()
    meal_master.gather_user_preferences()
    meal_plan = meal_master.generate_meal_plan()
    meal_master.display_meal_plan(meal_plan)# solution.py

import time
import random
from collections import deque

class CodeEfficiencyAnalyzer:
    """
    Analyzes the efficiency of the code by simulating a static code analysis.
    """
    def analyze_code(self, code_snippet):
        """
        Simulates code analysis and returns an efficiency score.
        :param code_snippet: A string representing a code snippet.
        :return: An efficiency score (float) between 0 and 1.
        """
        # Simulate a complex analysis process
        time.sleep(0.5)
        # Random efficiency score for demonstration purposes
        return random.uniform(0.5, 1.0)

class TaskManager:
    """
    Manages development tasks with dependencies.
    """
    def __init__(self):
        """
        Initializes the task manager with an empty task list and dependency graph.
        """
        self.tasks = {}
        self.dependencies = {}

    def add_task(self, task_id, dependencies=None):
        """
        Adds a task to the task manager.
        :param task_id: A unique identifier for the task.
        :param dependencies: A list of task IDs that this task depends on.
        """
        if dependencies is None:
            dependencies = []
        self.tasks[task_id] = False  # False indicates the task is not completed
        self.dependencies[task_id] = dependencies

    def get_task_order(self):
        """
        Returns the order in which tasks should be executed based on dependencies.
        :return: A list of task IDs in the order they should be executed.
        """
        in_degree = {task: 0 for task in self.tasks}
        for task, deps in self.dependencies.items():
            for dep in deps:
                in_degree[dep] += 1

        queue = deque([task for task, degree in in_degree.items() if degree == 0])
        order = []

        while queue:
            task = queue.popleft()
            order.append(task)
            for dep in self.dependencies[task]:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        if len(order) != len(self.tasks):
            raise ValueError("Cycle detected in task dependencies")

        return order

    def mark_task_complete(self, task_id):
        """
        Marks a task as completed.
        :param task_id: The ID of the task to mark as completed.
        """
        if task_id in self.tasks:
            self.tasks[task_id] = True

class BuildTimeEstimator:
    """
    Estimates the build time based on the number of tasks and their complexity.
    """
    def estimate_build_time(self, num_tasks, avg_task_complexity):
        """
        Estimates the build time.
        :param num_tasks: The number of tasks to be built.
        :param avg_task_complexity: The average complexity of the tasks.
        :return: An estimated build time in seconds.
        """
        # Simulate a build time estimation process
        time.sleep(0.5)
        # Random build time for demonstration purposes
        return num_tasks * avg_task_complexity * random.uniform(0.5, 1.5)

class CollaborativeBuildOptimizer:
    """
    The main class for the Collaborative Build Optimizer system.
    """
    def __init__(self):
        """
        Initializes the CBO system with the necessary components.
        """
        self.code_analyzer = CodeEfficiencyAnalyzer()
        self.task_manager = TaskManager()
        self.time_estimator = BuildTimeEstimator()

    def add_task(self, task_id, code_snippet, dependencies=None):
        """
        Adds a task to the system, analyzes its code efficiency, and estimates its build time.
        :param task_id: A unique identifier for the task.
        :param code_snippet: A string representing the code snippet for the task.
        :param dependencies: A list of task IDs that this task depends on.
        """
        efficiency_score = self.code_analyzer.analyze_code(code_snippet)
        print(f"Task {task_id} code efficiency score: {efficiency_score:.2f}")
        self.task_manager.add_task(task_id, dependencies)

    def get_task_order(self):
        """
        Returns the order in which tasks should be executed based on dependencies.
        :return: A list of task IDs in the order they should be executed.
        """
        return self.task_manager.get_task_order()

    def mark_task_complete(self, task_id):
        """
        Marks a task as completed.
        :param task_id: The ID of the task to mark as completed.
        """
        self.task_manager.mark_task_complete(task_id)

    def estimate_total_build_time(self):
        """
        Estimates the total build time for all tasks.
        :return: An estimated total build time in seconds.
        """
        num_tasks = len(self.task_manager.tasks)
        # Calculate the average task complexity by analyzing each task's code snippet
        avg_task_complexity = (
            sum(self.code_analyzer.analyze_code(code_snippet) for task_id, code_snippet in self.task_manager.tasks.items()) 
            / num_tasks
        )
        # Estimate the total build time using the number of tasks and average complexity
        return self.time_estimator.estimate_build_time(num_tasks, avg_task_complexity)

# Example usage
if __name__ == "__main__":
    cbo = CollaborativeBuildOptimizer()
    cbo.add_task("task1", "def foo(): return 42")
    cbo.add_task("task2", "def bar(): return foo() + 1", dependencies=["task1"])
    cbo.add_task("task3", "def baz(): return bar() * 2", dependencies=["task2"])

    task_order = cbo.get_task_order()
    print("Task execution order:", task_order)

    for task in task_order:
        print(f"Executing {task}...")
        time.sleep(1)  # Simulate task execution time
        cbo.mark_task_complete(task)
        print(f"{task} completed.")

    total_build_time = cbo.estimate_total_build_time()
    print(f"Estimated total build time: {total_build_time:.2f} seconds")

# The task description is: Write a program called MealMaster that personalizes meal plans for users based on their dietary preferences, health goals, and available ingredients. The program should take into account specific dietary needs such as calorie intake, macronutrient ratios, and food allergies, and suggest recipes with nutritional information for each meal. Based on this task description, I have improved the solution.