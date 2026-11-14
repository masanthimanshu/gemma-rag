# Gemma RAG AI Server

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LiteRT-LM](https://img.shields.io/badge/LiteRT--LM-0.19.1-orange.svg)](https://github.com/google-ai-edge/litert)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A high-performance, lightweight local AI inference and embedding server built with **FastAPI** and Google's **LiteRT-LM** runtime. Designed specifically for low-latency Retrieval-Augmented Generation (RAG) pipelines, on-device reasoning, and interactive chat streaming.

---

## Table of Contents

- [What the Project Does](#what-the-project-does)
- [Why the Project Is Useful](#why-the-project-is-useful)
- [Project Architecture](#project-architecture)
- [How Users Can Get Started](#how-users-can-get-started)
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
- [Where Users Can Get Help](#where-users-can-get-help)
- [Who Maintains and Contributes](#who-maintains-and-contributes)
  - [Maintainer](#maintainer)
  - [Contributing](#contributing)
  - [License](#license)

---

## What the Project Does

The **Gemma RAG AI Server** provides a self-hosted, private backend serving Google Gemma models directly on local hardware without cloud API dependencies. It delivers two foundational capabilities:

1. **Instruction & Reasoning Inference**: Executes chat queries using the `gemma-4-E2B-it` model with adaptive speculative decoding.
2. **Dense Vector Embeddings**: Generates semantic embeddings using the `embeddinggemma-2-740m` model for vector search, document indexing, and similarity retrieval in RAG systems.

All endpoints are exposed over asynchronous HTTP and WebSocket protocols powered by FastAPI and Uvicorn.

---

## Why the Project Is Useful

- **100% Local & Private**: All inference and embeddings run entirely on-device, ensuring zero data leakage and complete data sovereignty.
- **Speculative Decoding Acceleration**: Configured with LiteRT-LM's adaptive speculative decoding for lower latency token generation.
- **Real-Time Token Streaming**: Built-in WebSocket streaming emits response tokens as they are generated.
- **Unified RAG Stack**: Serves generative language models and dense embedding models from a single lightweight runtime process.
- **Hardware-Optimized Throughput**: Utilizes compiled LiteRT-LM binaries with automated XNNPack kernel caching for efficient CPU and edge execution.

---

## Project Architecture

```text
ai-server/
├── main.py                          # Server entrypoint and FastAPI lifespan management
├── requirements.txt                 # Project dependencies
├── models/                          # LiteRT-LM model binaries and XNNPack caches
│   ├── gemma-4-E2B-it.litertlm
│   └── embeddinggemma-2-740m.litertlm
├── routes/                          # API endpoints and route definitions
│   ├── __init__.py                  # Central API router aggregator
│   ├── chat/
│   │   └── chat_routes.py           # POST /chat/response & WebSocket /chat/stream
│   ├── embedding/
│   │   └── embedding_routes.py      # POST /embedding/create
│   └── health/
│       └── health_routes.py         # GET /health/
├── schema/
│   ├── __init__.py                  # Schema exports
│   └── schemas.py                   # Pydantic request & validation models
└── services/
    ├── __init__.py                  # Service package interface
    └── llm/
        └── llm_service.py           # LiteRT-LM engine lifecycle, chat, & embeddings
```

---

## How Users Can Get Started

### Prerequisites

- **Python**: Version 3.10 or later
- **Operating System**: Linux (recommended) or macOS / Windows with C/C++ runtime support
- **Hardware**: At least 8 GB RAM and 10 GB free disk space for model weights

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

Place your pre-compiled LiteRT-LM model weights in the `models/` directory:

```bash
mkdir -p models
```

Copy the required model files into `models/`:
- `models/gemma-4-E2B-it.litertlm`: Conversational instruction-tuned model.
- `models/embeddinggemma-2-740m.litertlm`: Text embedding model.

> **Note**: Large model binaries and auto-generated XNNPack cache files (`*.xnnpack_cache`) are ignored by Git via `.gitignore`.

### Running the Server

Start the application with Python:

```bash
python main.py
```

Or run via the Uvicorn CLI directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 3500 --reload
```

The server listens on `http://0.0.0.0:3500`. Interactive OpenAPI documentation is accessible at:
- **Swagger UI**: `http://localhost:3500/docs`
- **ReDoc**: `http://localhost:3500/redoc`

---

## API Usage & Examples

### 1. Health Check

Verifies server availability and readiness.

- **Endpoint**: `GET /health/`
- **Response**:

```json
{
  "status": 200,
  "message": "It's Working!"
}
```

```bash
curl -X GET http://localhost:3500/health/
```

---

### 2. Synchronous Chat Inference

Submits a prompt and returns the generated answer alongside internal chain-of-thought reasoning.

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
  "status": 200,
  "message": "The sky appears blue because of Rayleigh scattering..."
}
```

---

### 3. Real-Time Streaming Chat (WebSocket)

Streams response tokens incrementally over a persistent WebSocket connection.

- **Endpoint**: `ws://localhost:3500/chat/stream`
- **Payload Protocol**: Send a plain-text prompt; receive streaming JSON message events.

#### Event Payloads

- **Response token**:
  ```json
  {"type": "response", "content": "Hello"}
  ```

#### Python Streaming Client Example

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
                if data["type"] == "response":
                    print(data["content"], end="", flush=True)
            except websockets.exceptions.ConnectionClosed:
                break

asyncio.run(stream_chat())
```

---

### 4. Text Embeddings

Generates dense floating-point vector representations for search queries or document chunks.

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
  "status": 200,
  "message": [
    0.0241,
    -0.0135,
    0.0589
  ]
}
```

---

## Configuration

Engine parameters and decoding settings can be configured in [`services/llm/llm_service.py`](services/llm/llm_service.py):

| Parameter | Default | Location | Description |
| :--- | :--- | :--- | :--- |
| `max_output_tokens` | `600` | `create_conversation` | Maximum output token limit per completion |
| `mode` | `ADAPTIVE` | `speculative_config` | LiteRT-LM speculative decoding mode |
| `port` | `3500` | `main.py` | Network port for the FastAPI server |

---

## Where Users Can Get Help

- **Issue Tracker**: Submit bug reports and feature requests via [GitHub Issues](https://github.com/masanthimanshu/gemma-rag/issues).
- **Interactive Documentation**: Explore OpenAPI documentation directly on a running server at `http://localhost:3500/docs`.
- **FastAPI Documentation**: Consult the [FastAPI Official Documentation](https://fastapi.tiangolo.com/) for routing and deployment guidance.
- **Google LiteRT**: Learn more about on-device AI runtime optimization at [Google LiteRT](https://github.com/google-ai-edge/litert).

---

## Who Maintains and Contributes

### Maintainer

- **Himanshu Masant** ([@masanthimanshu](https://github.com/masanthimanshu)) - [masanthimanshu@gmail.com](mailto:masanthimanshu@gmail.com)

### Contributing

Contributions, feedback, and pull requests are welcome:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'feat: add amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
