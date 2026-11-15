import litert_lm

llm_engine = None
embedding_engine = None

speculative_config = litert_lm.SpeculativeDecodingConfig(
    mode=litert_lm.SpeculativeDecodingMode.ADAPTIVE,
)


def init_engine():
    global llm_engine, embedding_engine

    llm_engine = llm_engine or litert_lm.Engine(
        "./models/gemma-4-E2B-it.litertlm",
        speculative_decoding_config=speculative_config,
    )

    embedding_engine = embedding_engine or litert_lm.EmbeddingEngine(
        "./models/embeddinggemma-2-740m.litertlm"
    )


def close_engine():
    global llm_engine, embedding_engine

    for engine in (llm_engine, embedding_engine):
        if engine:
            engine.close()

    llm_engine = embedding_engine = None


def get_conversation(think: bool = False):
    config = litert_lm.ThinkingConfig(thinking_token_budget=300) if think else None
    return llm_engine.create_conversation(max_output_tokens=1200, thinking_config=config)


def chat(message: str, think: bool = False):
    with get_conversation(think) as conv:
        return {"message": str(conv.send_message(message))}


def stream_chat(message: str, think: bool = False):
    with get_conversation(think) as conv:
        for chunk in conv.send_message_async(message):
            if text := str(chunk):
                yield text


def embedding(message: str):
    return embedding_engine.compute_embedding(message).embedding
