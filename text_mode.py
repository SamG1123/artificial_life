"""Text-only runner for quick command testing.

Usage:
    python text_mode.py
"""

from __future__ import annotations

from agent import AgentController


def main() -> None:
    agent = AgentController()

    # Disable voice-listening path for text-only testing.
    agent.ears.sleep = True
    agent.start()

    print("Text mode started. Type commands and press Enter.")
    print("Special commands: /exit, /quit, /status")

    try:
        while True:
            msg = input("you> ").strip()
            if not msg:
                continue

            if msg.lower() in {"/exit", "/quit"}:
                break

            if msg.lower() == "/status":
                st = agent.status()
                print(
                    f"state={st.get('brain_state')} mood={st.get('mood')} "
                    f"goal={st.get('current_goal') or 'None'}"
                )
                continue

            agent.send_message(msg)
    except KeyboardInterrupt:
        pass
    finally:
        agent.stop()


if __name__ == "__main__":
    main()
