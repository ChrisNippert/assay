ollamui
=======

Find, download, tune and chat with AI models that run on your own machine.

Run ./ollamui and open http://localhost:8484 (add -open to open it for
you). The first time, go to Settings and press "Install llama.cpp" (or put
your own llama-server on PATH), then download a model from Discover.

Data lives in ~/.local/share/ollamui (chats, models) and
~/.config/ollamui (presets, settings). Ollama's models in ~/.ollama/models
are picked up automatically. Editors can use http://localhost:8484/v1.

ollamui -help lists the options.
