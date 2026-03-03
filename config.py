from queue import Queue

global_command_queue = Queue(10)   # TTS responses
global_ss_queue = Queue(1)          # screenshots
global_goal_queue = Queue(10)       # PC_CONTROL goals for the executor