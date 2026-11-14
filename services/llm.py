import litert_lm

llm_engine = None
embedding_engine = None

thinking_config = litert_lm.ThinkingConfig(
    enable_thinking=True,
    thinking_token_budget=200,
)

speculative_config = litert_lm.SpeculativeDecodingConfig(
    mode=litert_lm.SpeculativeDecodingMode.ADAPTIVE,
)


def init_engine():
    global llm_engine, embedding_engine

    if llm_engine is None:
        llm_engine = litert_lm.Engine(
            "./models/gemma-4-E2B-it.litertlm",
            speculative_decoding_config=speculative_config,
        )

    if embedding_engine is None:
        embedding_engine = litert_lm.EmbeddingEngine(
            "./models/embeddinggemma-2-740m.litertlm"
        )


def close_engine():
    global llm_engine, embedding_engine

    if llm_engine is not None:
        llm_engine.close()
        llm_engine = None

    if embedding_engine is not None:
        embedding_engine.close()
        embedding_engine = None


def chat(message: str):
    init_engine()

    with llm_engine.create_conversation(
        thinking_config=thinking_config,
        max_output_tokens=600,
    ) as conversation:
        response = conversation.send_message(message)
        return {
            "response": str(response),
            "thinking": response.channels.get("thought"),
        }


def stream_chat(message: str):
    init_engine()

    with llm_engine.create_conversation(
        thinking_config=thinking_config,
        max_output_tokens=600,
    ) as conversation:
        for chunk in conversation.send_message_async(message):
            if thought := chunk.channels.get("thought"):
                yield {"type": "thinking", "content": thought}
            if text := str(chunk):
                yield {"type": "response", "content": text}


def embedding(message: str):
    init_engine()

    response = embedding_engine.compute_embedding(message)
    return response.embedding
