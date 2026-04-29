# solution.py

class Task:
    def __init__(self, name, priority, time_slots, status='pending'):
        self.name = name
        self.priority = priority
        self.time_slots = time_slots        self.dependencies = []self.status = status

    def update_status(self, new_status):
        self.status = new_status


class Meeting:
    def __init__(self, participants, time, agenda):
        self.participants = participants
        self.time = time
        self.agenda = agenda


class Project:
    def __init__(self, name):
        self.name = name
        self.tasks = []
        self.meetings = []
self.task_dependencies = {}
        self.progress = 0

    def add_task(self, task):
        self.tasks.append(task)
self.task_dependencies[task.name] = task.dependencies

    def add_meeting(self, meeting):
        self.meetings.append(meeting)

    def update_progress(self, new_progress):
        self.progress = new_progress


class User:
    def __init__(self, name, availability):
        self.name = name
        self.availability = availability
        self.tasks = []

    def add_task(self, task):
        self.tasks.append(task)

    def update_availability(self, new_availability):
        self.availability = new_availability


class Notification:
    def __init__(self, message, recipient):
        self.message = message
        self.recipient = recipient

    def send(self):
        print(f"Notification sent to {self.recipient}: {self.message}")


class Report:
    def __init__(self, data):
        self.data = data

    def generate(self):
        # Placeholder for report generation logic
        return f"Report generated with data: {self.data}"


class TeamSyncPro:
    def __init__(self):
        self.users = []
        self.projects = []
        self.notifications = []

    def add_user(self, user):
        self.users.append(user)

    def add_project(self, project):
        self.projects.append(project)

    def send_notification(self, notification):
        self.notifications.append(notification)
        notification.send()

    def generate_report(self, data):
        report = Report(data)
        return report.generate()


# Example usage
if __name__ == "__main__":
    team_sync_pro = TeamSyncPro()

    # Adding users
    user1 = User("Alice", ["Monday", "Wednesday", "Friday"])
    user2 = User("Bob", ["Tuesday", "Thursday", "Saturday"])
    team_sync_pro.add_user(user1)
    team_sync_pro.add_user(user2)

    # Adding projects
    project1 = Project("Project A")
    task1 = Task("Task 1", 1, ["Monday 10:00-12:00"])
    task2 = Task("Task 2", 2, ["Tuesday 14:00-16:00"])
    project1.add_task(task1)
    project1.add_task(task2)
    team_sync_pro.add_project(project1)

    # Sending notifications
    notification1 = Notification("Meeting scheduled for Monday 10:00", "Alice")
    team_sync_pro.send_notification(notification1)

    # Generating reports
    report_data = {"project": "Project A", "progress": 50}
    report = team_sync_pro.generate_report(report_data)
    print(report)
