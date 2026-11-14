# Gemma RAG AI Server

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LiteRT-LM](https://img.shields.io/badge/LiteRT--LM-0.19.1-orange.svg)](https://github.com/google-ai-edge/litert)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A high-performance, lightweight local AI inference and embedding server built with **FastAPI** and Google's **LiteRT-LM** engine. Designed specifically for low-latency Retrieval-Augmented Generation (RAG) pipelines, on-device reasoning, and interactive chat experiences.

---

## Table of Contents

- [What the Project Does](#what-the-project-does)
- [Why It Is Useful](#why-it-is-useful)
- [Project Architecture](#project-architecture)
- [How to Get Started](#how-to-get-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Model Preparation](#model-preparation)
  - [Running the Server](#running-the-server)
- [API Usage & Examples](#api-usage--examples)
  - [1. Health Check](#1-health-check)
  - [2. Synchronous Chat Inference](#2-synchronous-chat-inference)
  - [3. Real-Time Streaming Chat (WebSocket)](#3-real-time-streaming-chat-websocket)
  - [4. Text Embeddings](#4-text-embeddings)
- [Configuration](#configuration)
- [Where to Get Help](#where-to-get-help)
- [Maintainers & Contributing](#who-maintains-and-contributes)

---

## What the Project Does

The **Gemma RAG AI Server** provides an OpenAI-alternative local backend serving Google Gemma models directly on your hardware. It handles both:

1. **Instruction & Reasoning Inference**: Executes chat queries using the `gemma-4-E2B-it` model with adaptive speculative decoding and native chain-of-thought (thinking) outputs.
2. **Dense Vector Embeddings**: Generates embeddings using the `embeddinggemma-2-740m` model for vector search, document indexing, and semantic similarity matching in RAG systems.

All endpoints are exposed over high-performance HTTP and WebSocket protocols powered by FastAPI and Uvicorn.

---

## Why It Is Useful

- **100% Local & Private**: All inference and embeddings run locally without sending data to third-party cloud APIs.
- **Speculative Decoding Acceleration**: Configured with LiteRT-LM's adaptive speculative decoding for significant speedups in token generation latency.
- **Native Chain-of-Thought ("Thinking") Support**: Separates internal reasoning (`thought` channel) from user-facing responses, enabling structured introspection and explainability.
- **Real-Time Token Streaming**: Built-in WebSocket streaming transmits both thinking chunks and response tokens as they are produced.
- **Ready for RAG**: Houses both the generative LLM and the embedding model under one unified, efficient server process.
- **Resource Efficient**: Uses LiteRT-LM compiled models with XNNPack kernel caching for maximum CPU/edge throughput.

---

## Project Architecture

```text
ai-server/
├── main.py                     # Server entrypoint and FastAPI lifespan management
├── requirements.txt            # Python dependencies
├── models/                     # LiteRT-LM binary models and XNNPack caches
│   ├── gemma-4-E2B-it.litertlm
│   └── embeddinggemma-2-740m.litertlm
├── routes/                     # API routes and controllers
│   ├── __init__.py             # Main router registry
│   ├── chat/
│   │   └── chat_routes.py      # POST /chat/response & WebSocket /chat/stream
│   ├── embedding/
│   │   └── embedding_routes.py # POST /embedding/create
│   └── health/
│       └── health_routes.py    # GET /health
├── schema/
│   ├── __init__.py
│   └── schemas.py              # Pydantic request models
└── services/
    ├── __init__.py
    └── llm.py                  # LiteRT-LM engine lifecycle, chat, & embeddings
```

---

## How to Get Started

### Prerequisites

- **Python**: Version 3.10 or later (tested on 3.10–3.15)
- **C/C++ Build Environment**: Required for LiteRT / XNNPack dependencies on Linux
- **RAM/Storage**: At least 8 GB RAM and 10 GB free disk space for model weights

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/masanthimanshu/gemma-rag.git
   cd gemma-rag
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Model Preparation

Ensure your LiteRT-LM model files are placed in the `models/` directory:

- `models/gemma-4-E2B-it.litertlm`: Gemma instruct model for conversational inference.
- `models/embeddinggemma-2-740m.litertlm`: Gemma embedding model for vector computation.

> **Note**: Model files and XNNPack caches (`*.xnnpack_cache`) are excluded from version control via `.gitignore`.

### Running the Server

Start the application with Uvicorn:

```bash
python main.py
```

Or run directly with Uvicorn CLI:

```bash
uvicorn main:app --host 0.0.0.0 --port 3500 --reload
```

The server binds to `http://0.0.0.0:3500`. Interactive OpenAPI documentation will be accessible at:
- **Swagger UI**: `http://localhost:3500/docs`
- **ReDoc**: `http://localhost:3500/redoc`

---

## API Usage & Examples

### 1. Health Check

Verifies server status and availability.

- **Endpoint**: `GET /health`
- **Response**:

```json
{
  "message": "It's Working!"
}
```

```bash
curl -X GET http://localhost:3500/health
```

---

### 2. Synchronous Chat Inference

Generates an answer along with the internal chain-of-thought reasoning.

- **Endpoint**: `POST /chat/response`
- **Request Body**:

```json
{
  "message": "Why is the sky blue?"
}
```

- **Example Request**:

```bash
curl -X POST http://localhost:3500/chat/response \
  -H "Content-Type: application/json" \
  -d '{"message": "Why is the sky blue?"}'
```

- **Example Response**:

```json
{
  "response": "The sky appears blue because of Rayleigh scattering...",
  "thinking": "The user is asking about a basic physical phenomenon. Explain Rayleigh scattering concisely..."
}
```

---

### 3. Real-Time Streaming Chat (WebSocket)

Streams tokens and thinking status incrementally over a WebSocket channel.

- **Endpoint**: `ws://localhost:3500/chat/stream`
- **Protocol**: Send plain text messages, receive JSON frames.

#### Event Formats

- **Thinking step**:
  ```json
  {"type": "thinking", "content": "Analyzing input..."}
  ```
- **Response token**:
  ```json
  {"type": "response", "content": "Hello"}
  ```

#### Python WebSocket Client Example

```python
import asyncio
import json
import websockets

async def stream_chat():
    uri = "ws://localhost:3500/chat/stream"
    async with websockets.connect(uri) as websocket:
        await websocket.send("Explain quantum computing in one sentence.")
        
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                if data["type"] == "thinking":
                    print(f"[Thinking] {data['content']}", end="", flush=True)
                elif data["type"] == "response":
                    print(data["content"], end="", flush=True)
            except websockets.exceptions.ConnectionClosed:
                break

asyncio.run(stream_chat())
```

---

### 4. Text Embeddings

Generates dense floating-point vector embeddings for input text.

- **Endpoint**: `POST /embedding/create`
- **Request Body**:

```json
{
  "message": "Semantic search query for document retrieval"
}
```

- **Example Request**:

```bash
curl -X POST http://localhost:3500/embedding/create \
  -H "Content-Type: application/json" \
  -d '{"message": "Semantic search query for document retrieval"}'
```

- **Example Response**:

```json
{
  "response": [
    0.0241,
    -0.0135,
    0.0589,
    ...
  ]
}
```

---

## Configuration

Inference parameters can be adjusted in [`services/llm.py`](services/llm.py):

| Parameter | Default | Location | Description |
| :--- | :--- | :--- | :--- |
| `enable_thinking` | `True` | `thinking_config` | Enables chain-of-thought generation |
| `thinking_token_budget` | `200` | `thinking_config` | Maximum token allocation for thinking |
| `max_output_tokens` | `600` | `create_conversation` | Cap on generated response tokens |
| `mode` | `ADAPTIVE` | `speculative_config` | Speculative decoding optimization mode |
| `port` | `3500` | `main.py` | Port on which the API server listens |

---

## Where to Get Help

- **Issues & Bug Reports**: Submit an issue on the [GitHub Issues](https://github.com/masanthimanshu/gemma-rag/issues) tracker.
- **FastAPI Documentation**: Consult the [FastAPI Official Docs](https://fastapi.tiangolo.com/) for questions on routing and deployment.
- **LiteRT Framework**: Learn more about model compilation and runtime optimizations at [Google LiteRT](https://github.com/google-ai-edge/litert).

---

## Who Maintains and Contributes

### Maintainer

- **Himanshu Masant** ([@masanthimanshu](https://github.com/masanthimanshu)) - [masanthimanshu@gmail.com](mailto:masanthimanshu@gmail.com)

### Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'feat: add amazing feature'`).
4. Push to your branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.
