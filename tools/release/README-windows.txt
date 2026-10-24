ollamui
=======

Find, download, tune and chat with AI models that run on your own PC.
Nothing is sent to the internet except model and llama.cpp downloads.

Running it
----------
1. Unzip this folder anywhere (e.g. Documents\ollamui).
2. Double-click ollamui.exe. A console window opens, and the app opens
   in your browser at http://localhost:8484.
   - Windows may say "Windows protected your PC" because the program
     isn't signed. Click "More info", then "Run anyway".
3. The first time, go to Settings and press "Install llama.cpp". It
   downloads the engine that runs the models (about 35 MB).
4. Open Discover, pick a model that says it fits your GPU, and download it.
   Then chat.

Keep the console window open while you use it. Close it to quit; that
also unloads the models from your GPU. Double-clicking ollamui.exe again
while it runs just opens the page.

Good to know
------------
- A graphics card helps a lot. llama.cpp uses it through Vulkan, which
  comes with current AMD, NVIDIA and Intel drivers. Without a GPU, models
  run on the processor, slowly.
- Models are large (2-10 GB each). Settings -> Models -> "Models folder"
  moves them to another drive.
- Your chats, models and settings live in:
    %LOCALAPPDATA%\ollamui   (chats, models)
    %APPDATA%\ollamui        (presets, settings)
  Delete those folders to remove everything.
- If you already use Ollama, its models are picked up automatically.
- Editors like Zed can use your models: point them at
  http://localhost:8484/v1 (an OpenAI-compatible API).
- Reading PDFs in project files needs pdftotext (from Poppler for
  Windows) on your PATH. Other file types work as they are.

Options (run from a terminal: ollamui.exe -help)
-------
  -listen 0.0.0.0:8484   reach it from your other devices (no password!)
  -open=false            don't open the browser
  -models D:\models      keep models in another folder
