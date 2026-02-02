import planner_model, app_control, browser_control, file_control, system_control


class AutomationExecutor:
    def __init__(self):
        self.planner = planner_model.Planner()
        self.app_controller = app_control.AppController()
        self.browser_controller = browser_control.BrowserController()
        self.file_controller = file_control.FileController()
        self.system_controller = system_control.SystemController()
    
    def execute_goal(self, goal: str):
        plan = self.planner.create_plan(goal)
        if not self.planner.validate_plan(plan):
            raise ValueError("Generated plan contains invalid actions.")
        
        for step in plan.get("steps", []):
            action = step["action"]
            target = step.get("target", "")
            self.execute_step(action, target)

    def execute_step(self, action: str, target: str):
        if action == "open_app":
            self.app_controller.open_app(target)
        elif action == "navigate":
            self.browser_controller.navigate(target)
        elif action == "search":
            self.browser_controller.search(target)
        elif action == "click":
            self.browser_controller.click(target)
        elif action == "type":
            self.browser_controller.type_text(target)
        elif action == "download":
            self.file_controller.download_file(target)
        elif action == "move_file":
            self.file_controller.move_file(target)
        elif action == "create_folder":
            self.file_controller.create_folder(target)
        elif action == "delete_file":
            self.file_controller.delete_file(target)
        elif action == "shutdown":
            self.system_controller.shutdown()
        else:
            raise ValueError(f"Unknown action: {action}")