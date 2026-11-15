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
        vision_backend=litert_lm.Backend.CPU(),
        max_num_images=1,
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


def content(message: str, image: bytes | None = None):
    if image:
        return litert_lm.Contents.of(
            litert_lm.Content.Text(message),
            litert_lm.Content.ImageBytes(image),
        )

    return message


def chat(message: str, image: bytes | None = None, think: bool = False):
    with get_conversation(think) as conv:
        return {"message": str(conv.send_message(content(message, image)))}


def stream_chat(message: str, image: bytes | None = None, think: bool = False):
    with get_conversation(think) as conv:
        for chunk in conv.send_message_async(content(message, image)):
            if text := str(chunk):
                yield text


def embedding(message: str):
    return embedding_engine.compute_embedding(message).embedding
