import os
from openai import AzureOpenAI
from dotenv import load_dotenv


load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)


def postprocess_4_1_4(asr_text: str, example_pairs: str) -> str:
    model = "gpt-4.1"

    if isinstance(example_pairs, str):
        parsed = []
        for line in example_pairs.splitlines():
            if "➜" in line:
                left, right = line.split("➜", 1)
            elif "\t" in line:
                left, right = line.split("\t", 1)
            elif "  " in line:
                left, right = line.split("  ", 1)
            else:
                continue
            parsed.append((left.strip(), right.strip()))
        example_pairs_list = parsed
    else:
        example_pairs_list = example_pairs

    if (
        not example_pairs_list
        or any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in example_pairs_list)
    ):
        raise ValueError(
            "`example_pairs` must be a list of (ASR, GT) tuples or a string "
            "containing lines formatted like 'ASR ➜ GT'."
        )

    def make_block(k, a, g):
        return (
            f"<PAIR-{k}-ASR output with error>\n{a.strip()}\n</PAIR-{k}-ASR output with error>\n"
            f"<PAIR-{k}-Ground Truth (Desired)>\n{g.strip()}\n</PAIR-{k}-Ground Truth (Desired)>\n"
        )

    example_block = "\n".join(
        make_block(i + 1, a, g) for i, (a, g) in enumerate(example_pairs_list)
    )

    playbook_user_prompt = """
You will receive parallel ASR / ground-truth sentence pairs from the same aphasic speaker.
Do NOT quote the sentences in your rules.

Write a numbered CORRECTION PLAYBOOK covering:

1. Keep-rules – words or patterns that must stay if they occur in ground-truth.
2. Delete-rules – tokens that should be removed when absent in ground-truth.
3. Substitute-rules – common ASR mistakes as "wrong→right" pairs.
4. Repetition handling – when to collapse duplicate tokens.
5. Confidence guardrail – "if ≤ 80 % sure a change is correct, leave it".
6. Output spec – final transcript must be plain text, no bullets/notes.

Return the playbook only.

### END EXAMPLES
""".strip()

    playbook_resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=2048,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a speech-error analyst who derives detailed rule-sets "
                    "from ASR / ground-truth pairs."
                ),
            },
            {"role": "user", "content": playbook_user_prompt},
        ],
    )
    playbook = playbook_resp.choices[0].message.content.strip()

    correction_user_prompt = f"""
Follow the CORRECTION PLAYBOOK below.
If you are <60 % confident about a change, leave the token unchanged.
Return the corrected transcript only. Do not add any preamble or explanation.

### CORRECTION PLAYBOOK
{playbook}

### General Decision Rules you must follow:
0. Disordered speech – never rewrite or "fix" ungrammatical patient language; keep it exactly as spoken.
1. Keep word order – never paraphrase, reorder, or add new words.
2. Fillers – delete pure hesitation noises:
uh, um, er, ah, hmm, mm, uh-huh, uh-uh.
3. Self-corrections – oh gosh | my god | i mean | sorry | i am sorry:
always keep exactly one copy when any of these self-correction cues occur.
4. Repetitions:
if the same word is repeated back-to-back at least 3 times, keep it exactly twice;
otherwise collapse adjacent duplicates to a single token.
5. Stutters – remove truncated fragments before a hyphen (e.g. thin-think → think).
6. Non-lexical noises – remove tokens such as whew, sigh, laugh, cough.
7. Digits → words – spell every numeral out in plain English (13 → thirteen).
8. Colloquial Contractions – expand colloquial contractions into standard forms.

### Examples
{example_block}

### RAW ASR
<ASR>
{asr_text.strip()}
</ASR>
### END
""".strip()

    corr_resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=4096,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an ASR post-processor. Apply the given playbook "
                    "exactly; do not add commentary."
                ),
            },
            {"role": "user", "content": correction_user_prompt},
        ],
    )

    return corr_resp.choices[0].message.content.strip()
