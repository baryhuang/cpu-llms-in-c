#!/usr/bin/env python3
"""Configure the Chatbox client for the local Qwen3.8-27B server.

Writes a custom OpenAI-compatible provider pointing at the local
serving shim into Chatbox's config.json and makes it the default chat
model, so the client works with zero manual setup. The schema matches
the open-source Chatbox settings types (settings.customProviders,
settings.providers, settings.defaultChatModel). If the app is running
it is quit first so it cannot overwrite the config on exit; the caller
reopens it. A .bak copy of the previous config is kept.
"""

import json
import os
import subprocess
import sys
import time

PROVIDER_ID = "qwen38-local"
MODEL_ID = "qwen3.8-27b"
CONFIG_PATH = os.path.expanduser(
    "~/Library/Application Support/xyz.chatboxapp.app/config.json")


def app_running():
    return bool(subprocess.run(["pgrep", "-x", "Chatbox"],
                               capture_output=True).stdout.strip())


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as handle:
            return json.load(handle)
    return {}


def configured(config, base_url):
    settings = config.get("settings", {})
    provider = settings.get("providers", {}).get(PROVIDER_ID, {})
    return (provider.get("apiHost") == base_url and
            any(model.get("modelId") == MODEL_ID
                for model in provider.get("models") or []) and
            any(entry.get("id") == PROVIDER_ID
                for entry in settings.get("customProviders") or []) and
            (settings.get("defaultChatModel") or {}).get("provider")
            == PROVIDER_ID)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else \
        "http://127.0.0.1:8199/v1"

    if configured(load_config(), base_url):
        print("Chatbox already configured for the local model")
        return 0

    if app_running():
        subprocess.run(["osascript", "-e", 'quit app "Chatbox"'],
                       capture_output=True)
        for _ in range(50):
            if not app_running():
                break
            time.sleep(0.2)

    config = load_config()
    settings = config.setdefault("settings", {})
    custom = settings.setdefault("customProviders", [])
    if not any(entry.get("id") == PROVIDER_ID for entry in custom):
        custom.append({"id": PROVIDER_ID, "name": "Qwen3.8-27B local",
                       "type": "openai", "isCustom": True})
    providers = settings.setdefault("providers", {})
    provider = providers.setdefault(PROVIDER_ID, {})
    provider["apiHost"] = base_url
    provider["apiKey"] = "local"
    provider["models"] = [{
        "modelId": MODEL_ID, "type": "chat", "apiStyle": "openai",
        "nickname": "Qwen3.8-27B (local C/Metal)",
        "contextWindow": 4096, "capabilities": [],
    }]
    settings["defaultChatModel"] = {"provider": PROVIDER_ID,
                                    "model": MODEL_ID}

    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as handle:
            previous = handle.read()
        with open(CONFIG_PATH + ".bak", "w") as handle:
            handle.write(previous)
    with open(CONFIG_PATH, "w") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    print(f"Chatbox configured: provider '{PROVIDER_ID}' at {base_url}, "
          f"default model '{MODEL_ID}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
