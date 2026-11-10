# deploy

Things that run the published models somewhere else.

| Folder | What |
|---|---|
| [space/](space/) | The [LiteGPT V2 Chat](https://huggingface.co/spaces/kmkrworks/LiteGPT-V2-Chat) Space: a static page that runs the Instruct Q4_K_M GGUF in the visitor's browser with [wllama](https://github.com/ngxson/wllama) 3.8.1. Published as is with `huggingface_hub` (`repo_type="space"`, `space_sdk="static"`) |
