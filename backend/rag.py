import os
import re

import torch
import chromadb

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

from dotenv import load_dotenv

load_dotenv(
    os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        ".env",
    )
)


# ============================================================
# CONFIGURATION
# (env values fall back to the previous hardcoded defaults)
# ============================================================

MODEL_NAME = os.getenv(
    "LLM_MODEL",
    "Qwen/Qwen2.5-0.5B-Instruct",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

_CHROMA_DIR = (
    os.getenv("CHROMA_DIR", "chroma_db").strip()
    or "chroma_db"
)

if not os.path.isabs(_CHROMA_DIR):
    _CHROMA_DIR = os.path.join(PROJECT_ROOT, _CHROMA_DIR)

CHROMA_PATH = _CHROMA_DIR
COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "thirdvizion",
)

# Retrieve only the most relevant chunks
TOP_K = int(os.getenv("TOP_K", "3"))

# Maximum generated answer tokens (short answers)
MAX_NEW_TOKENS = 62

# Minimum generated answer tokens (short answers).
# Direct answers come out as 2 to 3 lines.
NORMAL_MIN_TOKENS = 38

# Sentence stop floor (short answers).
# Stops at the first sentence end after this many
# tokens, giving 2 to 3 sentences per answer.
NORMAL_STOP_MIN_TOKENS = 48

# Maximum generated answer tokens
# (complete answers: lists and details)
LONG_MAX_TOKENS = 110

# Tokens reserved for the system prompt,
# question and template overhead.
# The retrieved context gets the remainder.
RESERVED_TOKENS = 740

# Context token budget for complete answers
LONG_CONTEXT_BUDGET = 460

# Questions asking for lists or details get
# broader retrieval and a bigger token budget.
LONG_ANSWER_RE = re.compile(
    r"\b(details?|detailed|explain|describe|"
    r"elaborate|elaboration|tell me more|"
    r"in depth|everything|what services|"
    r"which services|full list|complete list|"
    r"all (the )?services|"
    r"list (of |all |your )?services?)\b",
    re.IGNORECASE,
)

# Pure enumeration asks must not stop after a few
# items: force a minimum length for complete lists.
LIST_ANSWER_RE = re.compile(
    r"\b(what services|which services|"
    r"full list|complete list|"
    r"all (the )?services|"
    r"list (of |all |your )?services?)\b",
    re.IGNORECASE,
)

LIST_MIN_TOKENS = 70

# Service topics used to order chunks for complete
# answers so the services tables come first.
SERVICE_TOPICS = [
    "erp",
    "crm",
    "iam",
    "server",
    "web",
    "mobile",
    "app",
    "game",
    "marketing",
    "virtual",
    "augmented",
    "reality",
    "3d",
]


def _topic_set(text: str):

    low = text.lower()

    return {
        topic
        for topic in SERVICE_TOPICS
        if re.search(
            r"\b" + re.escape(topic) + r"\b",
            low,
        )
    }

# Maximum prompt tokens sent to the LLM.
# Keeps CPU inference fast (prefill cost grows
# quadratically with prompt length).
MAX_INPUT_TOKENS = 1024

# If the closest retrieved chunk is this far away
# (Chroma L2 distance) AND the question mentions
# none of the known service topics, answer with the
# standard fallback instead of asking the LLM.
# Prevents hallucinated answers to off-topic questions.
POOR_MATCH_DISTANCE = 1.45

SERVICE_TOPIC_RE = re.compile(
    r"\b(thirdvizion|third vizion|services?|"
    r"offers?|offering|provid\w*|compan(y|ies)|"
    r"mission|product|purpose|erp|crm|iam|"
    r"servers?|websites?|web|mobile|apps?|"
    r"games?|gaming|marketing|digital|virtual|"
    r"augmented|reality|cloud|software|develop\w*|"
    r"automat\w*|solutions?|technolog\w*|"
    r"pricing|prices?|costs?|contact|support|"
    r"business(es)?|clients?|projects?|teams?|"
    r"hir(e|ing)|quotes?|\bai\b|\bvr\b|\bar\b|3d)\b",
    re.IGNORECASE,
)

FALLBACK_MESSAGE = (
    "I can help you with ThirdVizion's "
    "services and related topics."
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the official ThirdVizion AI assistant.

Your job is to answer questions ONLY about ThirdVizion's services
and topics directly related to those services.

The provided document information is the primary source of truth
for ThirdVizion-specific information.

============================================================
ALLOWED TOPICS
============================================================

Answer questions related to ThirdVizion and its services, including
topics such as:

- ERP
- CRM
- IAM
- Server Management
- Web Development
- Mobile App Development
- Game Development
- Digital Marketing
- Virtual Reality
- Augmented Reality
- 3D Services
- Other services or technologies clearly related to ThirdVizion's
  provided information.

The list above is only an example. Use the provided document
information to determine whether the question is relevant.

============================================================
DOCUMENT INFORMATION
============================================================

When the provided documents contain relevant information:

- Use that information.
- Do not contradict it.
- Do not invent ThirdVizion-specific facts.
- Keep the answer concise.
- If the user asks for a list, use bullet points.
- If the user asks for details, explain the relevant information
  from the documents in a little more detail.

============================================================
WHEN DOCUMENT INFORMATION IS LIMITED
============================================================

If the question is about a general concept directly related to
a ThirdVizion service, you may give a short general explanation.

Example:

User:
"What is CRM?"

Answer:
"CRM (Customer Relationship Management) is a system used to
manage customer information, interactions, and relationships."

But do NOT claim that ThirdVizion specifically uses a particular
CRM platform unless that information is provided in the documents.

============================================================
UNRELATED QUESTIONS
============================================================

If the question is unrelated to ThirdVizion or its services,
do NOT answer it using general knowledge.

Examples of unrelated questions:

- What is Python?
- What is Java?
- What is Docker?
- What is mathematics?
- Who is the Prime Minister?
- What is the weather?

For unrelated questions, respond:

"I can help you with ThirdVizion's services and related topics."

============================================================
IDENTITY
============================================================

If the user asks:

"Who are you?"
"What are you?"
"What is your name?"

Answer:

"I am the ThirdVizion AI assistant. I can help you with
ThirdVizion's services and related information."

============================================================
GREETING
============================================================

For:

Hi
Hello
Hey
Good morning
Good afternoon
Good evening

Give a short friendly response.

Example:

"Hello! How can I help you with ThirdVizion's services?"

============================================================
STYLE
============================================================

- Keep answers short and natural.
- Do not repeat the user's question.
- Do not unnecessarily explain.
- Use bullet points for lists.
- Give more detail only when requested.
- Do not mention RAG.
- Do not mention ChromaDB.
- Do not mention embeddings.
- Do not mention vector databases.
- Do not mention chunks.
- Do not mention retrieval.
- Do not mention internal processing.
"""


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

print("Connecting to ChromaDB...")

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)

print("ChromaDB connected.")
print(f"Collection: {COLLECTION_NAME}")
print(f"Documents stored: {collection.count()}")


# ============================================================
# LOAD LLM
# ============================================================

print("Loading AI model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto"
)

model.eval()

print("AI model loaded.")


# ============================================================
# SENTENCE STOP
# (end prose answers at the first complete sentence
# so typical answers return faster; lists without
# sentence ends still use the full token budget)
# ============================================================

class SentenceStop(StoppingCriteria):

    def __init__(
        self,
        prompt_length: int,
        min_tokens: int = 26,
    ):

        self.prompt_length = prompt_length
        self.min_tokens = min_tokens

    def __call__(
        self,
        input_ids,
        scores,
        **kwargs,
    ) -> bool:

        new_ids = input_ids[0][self.prompt_length:]

        if len(new_ids) < self.min_tokens:

            return False

        text = tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
        )

        # A trailing list number is not a sentence end
        if re.search(r"(^|\n)\s*\d+\.\s*$", text):

            return False

        if re.search(
            r"[.!?]['\"]?\s*$",
            text,
        ) and not re.search(
            r"\b(e\.g|i\.e|u\.s|vs|etc)\.\s*$",
            text,
            flags=re.IGNORECASE,
        ):

            return True

        return False


# ============================================================
# STOP TOKENS
# ============================================================

EOS_IDS = [tokenizer.eos_token_id]

try:

    im_end_id = tokenizer.convert_tokens_to_ids(
        "<|im_end|>"
    )

    if (
        isinstance(im_end_id, int)
        and im_end_id not in EOS_IDS
    ):

        EOS_IDS.append(im_end_id)

except Exception:

    pass


# ============================================================
# MODEL DEVICE
# ============================================================

try:
    MODEL_DEVICE = next(model.parameters()).device
except Exception:
    MODEL_DEVICE = torch.device("cpu")


# ============================================================
# SIMPLE CONVERSATION
# ============================================================

def conversation_response(question: str):

    q = question.lower().strip()


    # --------------------------------------------------------
    # IDENTITY
    # --------------------------------------------------------

    if q in {
        "who are you",
        "what are you",
        "what is your name",
        "who is this"
    }:

        return (
            "I am the ThirdVizion AI assistant. "
            "I can help you with ThirdVizion's services "
            "and related information."
        )


    # --------------------------------------------------------
    # GREETINGS
    # --------------------------------------------------------

    if q in {
        "hi",
        "hello",
        "hey",
        "hai",
        "hi there",
        "hello there"
    }:

        return (
            "Hello! How can I help you with ThirdVizion's services?"
        )


    if q == "good morning":

        return (
            "Good morning! How can I help you with "
            "ThirdVizion's services?"
        )


    if q == "good afternoon":

        return (
            "Good afternoon! How can I help you with "
            "ThirdVizion's services?"
        )


    if q == "good evening":

        return (
            "Good evening! How can I help you with "
            "ThirdVizion's services?"
        )


    # --------------------------------------------------------
    # THANK YOU
    # --------------------------------------------------------

    if q in {
        "thanks",
        "thank you",
        "thankyou"
    }:

        return "You're welcome!"


    # --------------------------------------------------------
    # GOODBYE
    # --------------------------------------------------------

    if q in {
        "bye",
        "goodbye"
    }:

        return "Goodbye! Have a great day."


    return None


# ============================================================
# CLEAN MODEL OUTPUT
# ============================================================

META_LINE_RE = re.compile(
    r"^(if the (user asks|question)\b"
    r"|the user asked\b"
    r"|therefore\b.*\buser\b"
    r"|answering\b"
    r"|when the question\b"
    r"|when document information\b"
    r"|unrelated questions\b"
    r"|for\s*:?\s*$"
    r"|example\s*:?\s*$"
    r"|here'?s hoping\b"
    r"|the user might\b"
    r"|provide a short answer giving\b"
    r"|give a short friendly greeting\b"
    r"|(hi|hello|hey|hai|good morning|good afternoon|good evening|hello there|hi there)[,!.\"]*\s*$)",
    re.IGNORECASE,
)

# If the final text still looks like prompt/instruction
# echo instead of an answer, use the fallback.
META_JUNK_RE = re.compile(
    r"(explanation of answer"
    r"|no specific question being asked"
    r"|wikipedia"
    r"|do not write"
    r"|must end with"
    r"|repeat the question"
    r"|state your conclusion"
    r"|as per .*answer format"
    r"|information science and technology for\b)",
    re.IGNORECASE,
)


def clean_answer(answer: str, question: str = "") -> str:

    # Remove echoed prompt headers such as:
    # ANSWER, ===..., Answer:, Assistant:

    lines = answer.replace("\r", "").split("\n")

    # Normalized user question for echo detection
    question_norm = re.sub(
        r"[^a-z0-9 ]+",
        "",
        question.lower(),
    ).strip()

    kept = []

    for line in lines:

        stripped = line.strip()

        if not stripped:

            if kept:

                kept.append("")

            continue

        if set(stripped) <= set("= - * # > . [ ] ( )"):
            continue

        bare = re.sub(
            r"[\s\*#>=_\-:\"]+",
            "",
            stripped,
        ).lower()

        if bare in {
            "answer",
            "answers",
            "withanswer",
            "question",
            "greeting",
            "assistant",
            "ai",
            "response",
        }:
            continue

        if re.match(
            r"^\*{0,2}\s*question\s*\*{0,2}\s*:",
            stripped,
            flags=re.IGNORECASE,
        ):
            continue

        lead = re.sub(
            r"^[\s\*#>=_\-\"]+",
            "",
            stripped,
        )

        if re.match(
            r"question(\s*:|$)",
            lead,
            flags=re.IGNORECASE,
        ):
            continue

        if META_LINE_RE.match(lead):
            continue

        # Drop lines that only repeat the user question
        line_norm = re.sub(
            r"[^a-z0-9 ]+",
            "",
            stripped.lower(),
        ).strip()

        if (
            question_norm
            and line_norm
            and (
                line_norm == question_norm
                or line_norm.startswith(question_norm)
            )
        ):
            continue

        # Drop shouted prompt-header echoes
        # (e.g. WHEN DOCUMENT INFORMATION IS LIMITED)
        letters = re.sub(
            r"[^A-Za-z]",
            "",
            stripped,
        )

        if len(letters) >= 12 and letters.isupper():
            continue

        kept.append(stripped)

    text = "\n".join(kept).strip()

    # Collapse duplicate lines (model repetition loops).
    # Blanks are collapsed to a single blank line.

    deduped_lines = []
    seen_lines = set()

    for line in text.split("\n"):

        collapsed = re.sub(
            r"^(-\s+){2,}",
            "- ",
            line.strip(),
        )

        if not collapsed:

            if deduped_lines and not deduped_lines[-1]:

                continue

            deduped_lines.append("")

            continue

        if collapsed.lower() in seen_lines:

            continue

        seen_lines.add(collapsed.lower())
        deduped_lines.append(collapsed)

    text = "\n".join(deduped_lines).strip()

    # Drop a leading throat-clearing intro when the real
    # answer (list or details) follows after it, so the
    # token budget carries facts instead of filler.

    intro_lines = text.split("\n")

    if (
        len(intro_lines) > 1
        and re.match(
            r"^based on the (information|documentation|context|text|given)\b",
            intro_lines[0].strip(),
            flags=re.IGNORECASE,
        )
    ):

        text = "\n".join(intro_lines[1:]).strip()

    text = re.sub(
        r"^[\s\*#>=_\"]+",
        "",
        text,
    )

    label = re.match(
        r"(answer|answers|assistant|ai|response)(?=\s*:|\s*$)",
        text,
        flags=re.IGNORECASE,
    )

    if label:

        text = text[label.end():]

    text = re.sub(
        r"^[\s\*:]+",
        "",
        text,
    )

    # Drop meta-talk sentences inside paragraphs
    # (bullet lines are kept as-is)

    final_lines = []

    for line in text.split("\n"):

        if line.strip().startswith("-"):

            final_lines.append(line)

            continue

        sentences = re.split(
            r"(?<=[.!?])\s+",
            line,
        )

        kept_sentences = [
            sentence
            for sentence in sentences
            if sentence.strip()
            and not META_LINE_RE.match(sentence.strip())
        ]

        if kept_sentences:

            final_lines.append(" ".join(kept_sentences))

    text = "\n".join(final_lines).strip()

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(question: str) -> str:

    question = question.strip()

    if not question:

        return "Please enter a question."

    # List and detail questions get broader retrieval
    # and a bigger token budget for complete answers.

    is_long_answer = bool(
        LONG_ANSWER_RE.search(question)
    )

    is_list_answer = bool(
        LIST_ANSWER_RE.search(question)
    )


    # ========================================================
    # STEP 1
    # HANDLE GREETINGS / IDENTITY WITHOUT LLM
    # ========================================================

    simple_response = conversation_response(question)

    if simple_response:

        return simple_response


    # ========================================================
    # STEP 2
    # EMBED QUESTION
    # ========================================================

    question_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()


    # ========================================================
    # STEP 3
    # SEARCH CHROMADB
    # ========================================================

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=8 if is_long_answer else TOP_K,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )


    # ========================================================
    # STEP 4
    # GET DOCUMENTS
    # ========================================================

    documents = results.get(
        "documents",
        [[]]
    )[0]


    if not documents:

        return FALLBACK_MESSAGE


    # ========================================================
    # STEP 4b
    # OFF-TOPIC GUARD
    # (far retrieval match + no service topic mentioned:
    # use the documented fallback instead of guessing)
    # ========================================================

    try:

        all_distances = results.get(
            "distances",
            [[]]
        )[0] or []

        known = [
            dist
            for dist in all_distances
            if dist is not None
        ]

        best_distance = min(known) if known else None

    except Exception:

        best_distance = None

    if (
        best_distance is not None
        and best_distance >= POOR_MATCH_DISTANCE
        and not SERVICE_TOPIC_RE.search(question)
    ):

        return FALLBACK_MESSAGE


    # ========================================================
    # STEP 5
    # COMBINE RELEVANT DOCUMENTS
    # (trimmed to a token budget for fast inference.
    # Complete answers: densest chunks first so the
    # services tables fit inside the bigger budget.)
    # ========================================================

    # Drop bare URLs, site paths and table boilerplate
    # so the token budget carries facts.

    cleaned_docs = []

    for doc in documents:

        kept_lines = []

        for line in doc.split("\n"):

            stripped_line = line.strip()

            if not stripped_line:
                continue

            if re.match(
                r"^/[\w\-/#.]+$",
                stripped_line,
            ):
                continue

            if stripped_line.lower() in {
                "service",
                "description",
                "page",
            }:
                continue

            kept_lines.append(line)

        cleaned_docs.append("\n".join(kept_lines))

    if is_long_answer:

        # Greedy set cover: first the chunks adding
        # the most uncovered service topics, so the
        # services tables lead within the budget.

        scored = [
            (doc, _topic_set(doc))
            for doc in cleaned_docs
        ]

        ranked_docs = []
        covered_topics = set()
        pool = scored

        while pool:

            pool.sort(
                key=(
                    lambda item: (
                        len(item[1] - covered_topics),
                        len(item[1]),
                    )
                ),
                reverse=True,
            )

            if not pool[0][1] - covered_topics:

                if ranked_docs:

                    pool.sort(
                        key=(
                            lambda item: len(item[1])
                        ),
                        reverse=True,
                    )

                    ranked_docs.extend(
                        doc for doc, _ in pool
                    )

                    break

            best_doc, best_topics = pool.pop(0)

            ranked_docs.append(best_doc)
            covered_topics |= best_topics

        context_budget = LONG_CONTEXT_BUDGET

    else:

        ranked_docs = cleaned_docs

        context_budget = max(
            MAX_INPUT_TOKENS - RESERVED_TOKENS,
            100
        )

    context = "\n\n".join(ranked_docs)

    context = re.sub(
        r"https?://\S+",
        "",
        context,
    )

    context = re.sub(
        r"[ \t]+",
        " ",
        context,
    )

    context = re.sub(
        r"\n{3,}",
        "\n\n",
        context,
    ).strip()

    context_ids = tokenizer(
        context,
        add_special_tokens=False
    )["input_ids"][:context_budget]

    context = tokenizer.decode(
        context_ids,
        skip_special_tokens=True
    )


    # ========================================================
    # STEP 6
    # CREATE PROMPT
    # (chat template: the Instruct model answers
    # instead of extending prompt instructions)
    # ========================================================

    user_content = (
        "ThirdVizion information:\n"
        f"{context}\n"
        "\n"
        f"Question: {question}\n"
        + (
            "Answer completely. "
            "Use short bullet points for lists."
            if is_long_answer else
            "Answer in 2 to 3 short sentences."
        )
    )

    chat_text = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.strip(),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


    # ========================================================
    # STEP 7
    # TOKENIZE
    # (safety ceiling only; context is pre-trimmed above
    # so the question is never cut)
    # ========================================================

    inputs = tokenizer(
        chat_text,
        return_tensors="pt",
        truncation=True,
        max_length=1400 if is_long_answer else 1152
    )


    inputs = {
        key: value.to(MODEL_DEVICE)
        for key, value in inputs.items()
    }


    # ========================================================
    # STEP 8
    # GENERATE
    # ========================================================

    with torch.no_grad():

        if is_list_answer:

            min_tokens = LIST_MIN_TOKENS

        elif is_long_answer:

            min_tokens = 0

        else:

            min_tokens = NORMAL_MIN_TOKENS

        output = model.generate(
            **inputs,

            max_new_tokens=(
                LONG_MAX_TOKENS
                if is_long_answer else
                MAX_NEW_TOKENS
            ),

            min_new_tokens=min_tokens,

            do_sample=False,

            repetition_penalty=1.05,

            use_cache=True,

            pad_token_id=tokenizer.eos_token_id,

            eos_token_id=EOS_IDS,

            stopping_criteria=StoppingCriteriaList([
                SentenceStop(
                    inputs["input_ids"].shape[1],
                    min_tokens=(
                        80
                        if is_long_answer else
                        NORMAL_STOP_MIN_TOKENS
                    )
                )
            ])
        )


    # ========================================================
    # STEP 9
    # REMOVE INPUT TOKENS
    # ========================================================

    generated_tokens = output[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]


    # ========================================================
    # STEP 10
    # DECODE
    # ========================================================

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    answer = clean_answer(answer, question)


    # ========================================================
    # STEP 11
    # CLEAN ANSWER
    # ========================================================

    if not answer or META_JUNK_RE.search(answer):

        return FALLBACK_MESSAGE


    return answer


# ============================================================
# TERMINAL CHAT
# ============================================================

def main():

    print()
    print("=" * 60)
    print("ThirdVizion AI Assistant")
    print("=" * 60)

    print("Type your question.")
    print("Type 'exit' or 'quit' to stop.")

    print("=" * 60)


    while True:

        try:

            question = input("\nYou: ").strip()


            # ------------------------------------------------
            # EXIT
            # ------------------------------------------------

            if question.lower() in {
                "exit",
                "quit"
            }:

                print("\nGoodbye!")
                break


            # ------------------------------------------------
            # EMPTY
            # ------------------------------------------------

            if not question:

                continue


            # ------------------------------------------------
            # ANSWER
            # ------------------------------------------------

            print("\nAI: ", end="", flush=True)

            answer = ask_question(question)

            print(answer)


        except KeyboardInterrupt:

            print("\n\nGoodbye!")
            break


        except Exception as e:

            print("\nError:", str(e))


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
