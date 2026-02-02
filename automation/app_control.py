import os

class AppControl:
    def __init__(self):
        pass

    def normalize_app_name(self, app_name: str) -> str:
        return app_name.strip().lower()

    def open_app(self, app_name: str):
        app_name = self.normalize_app_name(app_name)
        if os.name == 'nt':  # Windows
            os.startfile(app_name)
        elif os.name == 'posix':  # macOS or Linux
            os.system(f'open -a "{app_name}"' if sys.platform == 'darwin' else f'{app_name} &')
        else:
            raise NotImplementedError("Unsupported OS")

    
if __name__ == "__main__":
    controller = AppControl()
    controller.open_app("notepad")