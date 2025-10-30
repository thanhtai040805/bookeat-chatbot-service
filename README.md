# Restaurant Chatbot Service

FastAPI-based backend that powers a restaurant concierge chatbot. The service can respond with handcrafted fallback answers or delegate to OpenAI when an API key is provided.

## Project Layout

```
ai-services/
├── app/
│   ├── RestaurantChatbot.py
│   └── __init__.py
├── Dockerfile
├── render.yaml
├── requirements.txt
└── README.md
```

`Dockerfile` and `render.yaml` live at the service root so Render (or any container platform) can pick them up without extra configuration.

## Local Development

1. Use Python 3.10+ and create an isolated virtual environment.
2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Run the API:
   ```bash
   uvicorn app.main:app --reload
   ```
4. Send a sample request:
   ```bash
   curl -X POST http://localhost:8000/chat ^
     -H "Content-Type: application/json" ^
     -d "{\"user_message\": \"Xin chao, toi muon dat ban toi nay.\"}"
   ```
5. Visit the interactive docs at `http://localhost:8000/docs`.

### Environment Variables

| Variable            | Required | Description                                                |
|---------------------|----------|------------------------------------------------------------|
| `OPENAI_API_KEY`    | No       | Enables OpenAI-powered responses when present.             |
| `OPENAI_MODEL`      | No       | Defaults to `gpt-4o-mini`.                                 |
| `OPENAI_TEMPERATURE`| No       | Creativity level; defaults to `0.4`.                       |

## Deploying on Render

1. Push the `ai-services` directory as its own repository (or set it as the repo root).
2. On Render, choose **New +** → **Blueprint** and select the repository.
3. Confirm the detected values from `render.yaml`:
   - Environment: `docker`
   - Build command: `pip install --upgrade pip && pip install -r requirements.txt`
   - Start command: `uvicorn app.RestaurantChatbot:app --host 0.0.0.0 --port $PORT`
4. Add `OPENAI_API_KEY` under the Render dashboard (Environment tab) if you want LLM answers.
5. Trigger a deploy. Render will use `/health` for checks.
6. Your chatbot endpoint lives at `/chat` on the generated Render URL.

## Docker Usage

Build and run locally to verify the image that Render will build:

```bash
docker build -t restaurant-chatbot .
docker run -p 8000:8000 restaurant-chatbot
```

The container uses the same start command as Render, so successful local tests give confidence that the Render build will succeed.
