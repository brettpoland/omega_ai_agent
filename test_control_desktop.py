import sys
import types

from agent import control_desktop

def test_control_desktop_automation(monkeypatch):
    calls = []

    class DummyImage:
        def save(self, path):
            calls.append(("save", path))

    def screenshot():
        return DummyImage()

    def size():
        return (100, 100)

    def moveTo(x, y):
        calls.append(("moveTo", x, y))

    def click():
        calls.append(("click",))

    def write(text):
        calls.append(("write", text))

    def press(key):
        calls.append(("press", key))

    dummy_module = types.SimpleNamespace(
        screenshot=screenshot,
        size=size,
        moveTo=moveTo,
        click=click,
        write=write,
        press=press,
    )
    monkeypatch.setitem(sys.modules, "pyautogui", dummy_module)
    result = control_desktop("click start and open settings")
    assert "Screenshot saved to" in result
    assert ("moveTo", 10, 90) in calls
    assert ("press", "enter") in calls
