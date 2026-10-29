import random

import numpy as np
import pandas as pd
import streamlit as st


NUCLEOTIDES = "ATGC"
GENE_SIZE = 20
SUBBLOCK = 5
BENEFICIAL_MOTIFS = ("GATTACA", "ATCG", "CCGG", "TATA", "GCGC")
HARMFUL_MOTIFS = ("AAAA", "TTTT", "CGAT", "GGGG")
FATHER_DEFAULT = (
    "ATGCGATTACAGGCTAGCTAGGCGCGATTACATATAGCGCCCGGATCGATCGTATAGGCCGCGATTACAAAATGCGATCGGCGCTATAGCCGGATCGATTACAGCTAGCTAGGCGATTACATATAGCGCGGATCGATCGTAGGCCCGGATTACAAATGCGCTAGCTATAGGCGATTACACCGGATCGATCGTATAGCGC"
)
MOTHER_DEFAULT = (
    "ATGCGGGGATCGATCGCTAGCGCGATCGATATAGCGCAAATTTTATCGATCGTATAGGCCTATAGATTACAGGGGTGCGATCGGCGCTATAGCCAAATCGATCGAGCTAGCTAGGCGATCGATATAGCGCGGGGGATCGTAGGCCAAATCGATAAAATGCGCTAGCTATAGGCGATCGACCAAATCGATCGTATAGCGC"
)


def validate_dna(sequence: str, label: str) -> str:
    sequence = "".join(sequence.upper().split())
    if not sequence:
        raise ValueError(f"{label} sequence cannot be empty.")
    invalid = sorted(set(sequence) - set(NUCLEOTIDES))
    if invalid:
        raise ValueError(f"{label} contains invalid characters: {', '.join(invalid)}")
    if len(sequence) < GENE_SIZE:
        raise ValueError(f"{label} must contain at least {GENE_SIZE} bases.")
    return sequence


def fitness_score(segment: str) -> float:
    score = sum(segment.count(motif) * 2 for motif in BENEFICIAL_MOTIFS)
    score -= sum(segment.count(motif) * 3 for motif in HARMFUL_MOTIFS)
    gc_content = (segment.count("G") + segment.count("C")) / len(segment)
    return score + (1 - abs(gc_content - 0.5)) * 2


def build_markov_chain(sequence: str, order: int = 2) -> dict:
    counts = {}
    for index in range(len(sequence) - order):
        context = sequence[index : index + order]
        nucleotide = sequence[index + order]
        counts.setdefault(context, {}).setdefault(nucleotide, 0)
        counts[context][nucleotide] += 1
    return {
        context: (list(next_counts), np.array(list(next_counts.values())) / sum(next_counts.values()))
        for context, next_counts in counts.items()
    }


def run_simulation(
    father_sequence: str,
    mother_sequence: str,
    mutation_rate: float,
    attempts: int,
    seed: int,
) -> tuple[pd.DataFrame, str, str, int, int]:
    usable_length = min(len(father_sequence), len(mother_sequence))
    usable_length -= usable_length % GENE_SIZE
    father_sequence = father_sequence[:usable_length]
    mother_sequence = mother_sequence[:usable_length]
    gene_count = usable_length // GENE_SIZE
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    markov_chain = build_markov_chain(father_sequence + mother_sequence)

    def recombine_gene(father_gene: str, mother_gene: str) -> str:
        child_blocks = []
        for start in range(0, GENE_SIZE, SUBBLOCK):
            father_block = father_gene[start : start + SUBBLOCK]
            mother_block = mother_gene[start : start + SUBBLOCK]
            scores = np.array([fitness_score(father_block), fitness_score(mother_block)])
            probabilities = np.exp(scores - scores.max())
            probabilities /= probabilities.sum()
            child_blocks.append(
                father_block if rng.choice((0, 1), p=probabilities) == 0 else mother_block
            )

        child = list("".join(child_blocks))
        for index in range(len(child)):
            if py_rng.random() >= mutation_rate:
                continue
            context = "".join(child[index - 2 : index]) if index >= 2 else ""
            if context in markov_chain:
                choices, probabilities = markov_chain[context]
                child[index] = rng.choice(choices, p=probabilities)
            else:
                child[index] = py_rng.choice(NUCLEOTIDES)
        return "".join(child)

    def best_child_gene(father_gene: str, mother_gene: str) -> str:
        candidates = [recombine_gene(father_gene, mother_gene) for _ in range(attempts)]
        best = max(candidates, key=fitness_score)
        better_parent = max((father_gene, mother_gene), key=fitness_score)
        return max((best, better_parent), key=fitness_score)

    rows = []
    naive_parts = []
    child_parts = []
    for index in range(gene_count):
        start = index * GENE_SIZE
        father_gene = father_sequence[start : start + GENE_SIZE]
        mother_gene = mother_sequence[start : start + GENE_SIZE]
        naive_gene = py_rng.choice((father_gene, mother_gene))
        child_gene = best_child_gene(father_gene, mother_gene)
        naive_parts.append(naive_gene)
        child_parts.append(child_gene)
        rows.append(
            {
                "Gene": index + 1,
                "Father": fitness_score(father_gene),
                "Mother": fitness_score(mother_gene),
                "Naive child": fitness_score(naive_gene),
                "Optimized child": fitness_score(child_gene),
            }
        )

    return pd.DataFrame(rows), "".join(naive_parts), "".join(child_parts), len(father_sequence), len(mother_sequence)


st.set_page_config(page_title="DNA Inheritance Simulator", page_icon="🧬", layout="wide")
st.title("DNA Inheritance Simulator")
st.caption("An educational motif-scoring and recombination simulation. This is not a clinical or biological prediction tool.")

with st.sidebar:
    st.header("Simulation settings")
    mutation_rate = st.slider("Mutation rate", min_value=0.0, max_value=0.3, value=0.05, step=0.01)
    attempts = st.slider("Candidates per gene", min_value=1, max_value=20, value=6)
    seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=42)

with st.form("dna_inputs"):
    left, right = st.columns(2)
    father_input = left.text_area("Father DNA", FATHER_DEFAULT, height=180)
    mother_input = right.text_area("Mother DNA", MOTHER_DEFAULT, height=180)
    submitted = st.form_submit_button("Run simulation", type="primary")

if submitted:
    try:
        father_sequence = validate_dna(father_input, "Father")
        mother_sequence = validate_dna(mother_input, "Mother")
    except ValueError as error:
        st.error(str(error))
    else:
        usable_length = min(len(father_sequence), len(mother_sequence))
        usable_length -= usable_length % GENE_SIZE
        if usable_length == 0:
            st.error(f"Both sequences need at least {GENE_SIZE} usable bases.")
        else:
            if len(father_sequence) != len(mother_sequence):
                st.info("Sequences have different lengths; the longer sequence is trimmed to match the shorter one.")
            if usable_length != min(len(father_sequence), len(mother_sequence)):
                st.info(f"The final {min(len(father_sequence), len(mother_sequence)) - usable_length} base(s) are omitted so genes are exactly {GENE_SIZE} bases.")

            fitness, naive_dna, child_dna, _, _ = run_simulation(
                father_sequence, mother_sequence, mutation_rate, attempts, int(seed)
            )
            father_total = fitness["Father"].sum()
            mother_total = fitness["Mother"].sum()
            naive_total = fitness["Naive child"].sum()
            child_total = fitness["Optimized child"].sum()
            changed = sum(first != second for first, second in zip(naive_dna, child_dna))

            st.subheader("Results")
            metric_columns = st.columns(4)
            metric_columns[0].metric("Father fitness", f"{father_total:.2f}")
            metric_columns[1].metric("Mother fitness", f"{mother_total:.2f}")
            metric_columns[2].metric("Naive child", f"{naive_total:.2f}")
            metric_columns[3].metric("Optimized child", f"{child_total:.2f}", f"{child_total - naive_total:+.2f} vs naive")

            st.line_chart(fitness.set_index("Gene"), y=["Father", "Mother", "Naive child", "Optimized child"])
            st.bar_chart(pd.Series(
                {"Father": father_total, "Mother": mother_total, "Naive child": naive_total, "Optimized child": child_total},
                name="Total fitness",
            ))

            st.subheader("DNA sequences")
            st.code(f">Naive_Child\n{naive_dna}\n>Optimized_Child\n{child_dna}", language="text")
            st.caption(f"Different positions between child sequences: {changed} / {len(child_dna)}")

            fasta = (
                f">Father\n{father_sequence[:usable_length]}\n"
                f">Mother\n{mother_sequence[:usable_length]}\n"
                f">Naive_Child\n{naive_dna}\n>Optimized_Child\n{child_dna}\n"
            )
            downloads = st.columns(2)
            downloads[0].download_button("Download FASTA", fasta, "dna_results.fasta", "text/plain")
            downloads[1].download_button("Download gene scores CSV", fitness.to_csv(index=False), "gene_fitness.csv", "text/csv")