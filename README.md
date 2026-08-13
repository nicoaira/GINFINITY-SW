# GINFINITY-SW

GINFINITY-SW is a standalone exact local Smith–Waterman aligner for ordered
vector sequences. It uses cosine similarity, a configurable affine score
transform, and affine gap costs. The native Numba path and the reference path
execute the same recurrence.

The package is independent of GINFINITY. It can align embeddings from any
encoder as long as both matrices have the same feature dimension.

## Install

From PyPI:

```bash
python -m pip install ginfinity-sw
```

From the tagged GitHub release:

```bash
python -m pip install \
  "git+https://github.com/nicoaira/GINFINITY-SW.git@v1.0.0"
```

Using conda:

```bash
conda install -c nicolas.aira -c conda-forge ginfinity-sw
```

For a local checkout:

```bash
python -m pip install .
```

## Python API

```python
import numpy as np
from ginfinity_sw import ScoringParameters, align

query = np.load("query.npy")       # shape (Lq, d)
target = np.load("target.npy")     # shape (Lt, d)
parameters = ScoringParameters(
    mu=0.3,
    gamma=4.0,
    gap_open=3.0,
    gap_extend=1.0,
    score_min=-6.0,
    score_max=8.0,
)

result = align(query, target, params=parameters)
print(result.score)
print(result.query_span, result.target_span)  # 0-based, half-open
print(result.columns)                          # -1 denotes a gap
```

Use the GINFINITY model-specific parameters without coupling the packages:

```python
from ginfinity import default_alignment_parameters
from ginfinity_sw import ScoringParameters

parameters = ScoringParameters(**default_alignment_parameters())
```

## Ranking

```python
from ginfinity_sw import rank

ranking = rank(
    query,
    [("candidate-a", candidate_a), ("candidate-b", candidate_b)],
    params=parameters,
)
```

Ranking is deterministic: descending score with identifier as the tie-breaker.

## Command line

The CLI consumes an NPZ archive keyed by transcript ID, a matching metadata TSV,
and a JSON scoring-parameter file:

Create the same `structures.tsv` used by GINFINITY. These are the complete
`AANU01100842.1/307-440` and `AAZO01007178.1/21512-21382` records from RF00548:

```text
transcript_id	sequence	secondary_structure
AANU01100842.1/307-440	CUUAUGAAGUCUUCCUUUCAGUUCAGAAGAAAUGGAAUUCGCUCUCCAACUUCAGGAAACUGAAAUAAAGAGUUGCUUGGAUUUAGUGUUCACCUUUACCAUAAAAUGGAUUUGCUAACACUGCCACCCUGCUUUGAUAGCGAAUAAAGCAAAAAGGGCUUCUGUCGUGAGUGGCACACGUAGGGCAACUCGAUUGCUCUUCGUGCGGAAUCGACAUCAAGAGAUUUCGGAAGCAUAAUUUUUUGACAUUCGGGCAGCUGGUGAUCGUUGGUCCCGGCGCCCUUCUUUUUUUCUGUCUCAAGUCAGAUGAAUUUUUCUGGUGAGUUAGGUGUUAGUUUUGUAAGUGGAUGUAAGAUUUAUGUUAAUCCUUUUUAUUUGAAGUUGCGUAGCUAUCUGCGUGAACCGCAGAUGACUAAAUUAGCAGGGUAUUUAAC	......((.((((..(((((((((.((((...((((........)))).))))..)).)))))))..)))).))....((...(((((((........((((...)))).......)))))))))((((((((......((((...(((.((((((((((((((((......))))((((.(((((((.....))))))).))))(((((((..........))))))))))))((((.((((..((((((((((.(((((.((((...))))))))))))).........((((.((((..(((((........))))))))).))))................)))))))))).))))......))))))).)))....)))).(((.((((((((.....)))))))).)))....)))))))).......
AAZO01007178.1/21512-21382	AUUCGGAAAAAAAUUUCAACGGAUAUAAAAUACGUUAAUUCAAAUCAUUUUAAACUUUAUCCGUUUUGAAAUAUUAUUUAGAGAUUUUAACCGAAGGAUUUAACGUUUAUGUAAAUUGUUAAAUGAAAGAAUAACCGUUUGAACUUUUAAUAAAAGGGUGCUUGCUGCUUAGAUAGCACAUAGGUCCAAACGGGCCUGUGGGGUAUGGUUAUCAAGACAUAUCCAGCACGUAAUUUUUGUACCUGUAGGGCUGUUGACUGCCAAUAUGCAUCGACGCCCUCGUAAUCGACCUCAAUGAUAAUAAUCUCUUAUACCUUUCGCACUUUCCACGUGUACCUGGCAAUUACAAUUUAAUUAUCGGGUGAGUAGGUUCUUUUUUUGGCAAUCGGGUUGCUAAUUAUUAUCCCCCCUGGAAGGUGUUAACUUCAUCC	.(((((...((((((((((((((((((((((..(((......))).)))))).....))))))))...............))))))))..)))))..((((((((.((.....)).))))))))(((((..............(((((....)))))((((((((((......)))))(((((((((....)))))))))(((((((..........))))))))))))..................((((.(((((.(((......))))))))))))(((....))).......(((....)))........)))))((((((((((....(((((((.(((((.....))))).)))))))....((........((((((((...)))))))).......))....))))))))))...........
```

Generate embeddings and the model-specific scoring parameters with GINFINITY,
then align the pair:

```bash
ginfinity embed \
  --input structures.tsv \
  --output embeddings.npz

ginfinity alignment-config --output alignment.json

ginfinity-sw \
  --embeddings embeddings.npz \
  --metadata structures.tsv \
  --parameters alignment.json \
  --query "AANU01100842.1/307-440" \
  --target "AAZO01007178.1/21512-21382" \
  --output result.json

jq -r .formatted result.json
```

Omit `--target` to rank the query against every other array in the archive.

### Resulting alignment

Using the bundled GINFINITY 1.0 encoder and its exported scoring parameters,
the commands above produce:

```text
Score = 52.4872
Aligned columns = 138 (127 ungapped); base identity = 57.5%; structure identity = 78.0%; conserved pairs = 36
Query span = (151, 285); Target span = (151, 282) (0-based, half-open)

Query   152 AAAAGGGCUUCUGUCGUGA--GUGGCACACGUAGGGCAACUCGAUUGCUCUUCGUGCGGA 209
            (((((((((((((((....--..))))((((.(((((((.....))))))).))))((((
                  <<<<<<<<<        >>>><<<   <<<<<       >>>>>   >>><<<<
            ::::::|++++||++:~~~  ~:+|||||+  |||+|+:~~|   +|++|  +|||+||+
            .)))))((((((((((......)))))(((--((((((....---)))))--))))((((
Sbjct   152 AAAAGGGUGCUUGCUGCUUAGAUAGCACAU--AGGUCCAAAC---GGGCC--UGUGGGGU 204

Query   210 AUCGACAUCAAGAGAUUUCGGAAGCAUAAUUUUUUGACA-UUCGGGCAGCUGGUGAUCGU 268
            (((..........))))))))))))((((.((((..(((-(((((((.(((((.((((..
            <<<          >>>>>>>>>>>>                  <<<< <<<<< <<<
            ||+|~~|||||||+||+||+++++|.:::|::::~~::. .:.||||~|+||+~+++:~~
            (((..........))))))))))))..................((((.(((((.(((...
Sbjct   205 AUGGUUAUCAAGACAUAUCCAGCACGUAAUUUUUGUACCUGUAGGGCUGUUGACUGCCAA 264

Query   269 UG-GUCCCGGCGCCCUUC 285
            .)-))))))))))))...
               >>>>>>>>>>>>
            |. |+++||+|||||:..
            ...))))))))))))(((
Sbjct   265 UAUGCAUCGACGCCCUCG 282
```

### Reading the alignment

Each block contains six biological rows:

1. Query sequence, with 1-based inclusive coordinates.
2. Query dot-bracket structure.
3. Conserved-pair markers: `<` and `>` mark query pair endpoints that align to
   a base pair in the subject.
4. Match symbols describing each aligned column.
5. Subject dot-bracket structure.
6. Subject sequence, with 1-based inclusive coordinates.

The summary spans above the blocks use Python-style 0-based, half-open
coordinates. A `-` within a sequence or structure row is an alignment gap.

| symbol | base relationship | structure relationship |
|---|---|---|
| `|` | same base | same state |
| `+` | different base | both paired on the same side (`(` with `(` or `)` with `)`) |
| `~` | different base | both unpaired |
| `:` | same base | different state |
| `.` | different base | different state |
| space | gap | gap |

“State” distinguishes opening-pair `(`, unpaired `.`, and closing-pair `)`.
Consequently, `+` reports structurally corresponding paired positions despite
a nucleotide substitution, whereas `~` is agreement between unpaired sites.

### Custom metadata columns

The metadata reader accepts arbitrary column names, order, extra columns, and
delimiter. For `name,bases,dot_bracket,source`, use:

```bash
ginfinity-sw \
  --embeddings embeddings.npz \
  --metadata structures.csv \
  --parameters alignment.json \
  --query query-id \
  --target subject-id \
  --id-column name \
  --sequence-column bases \
  --structure-column dot_bracket \
  --delimiter ,
```

The equivalent Python reader is:

```python
from ginfinity_sw import read_metadata_table

metadata = read_metadata_table(
    "structures.csv",
    identifier_column="name",
    sequence_column="bases",
    structure_column="dot_bracket",
    delimiter=",",
)
```

## Resource limits

Exact alignment uses `Lq × Lt` dynamic-programming cells. The API and CLI
default to 16,777,216 cells per pair and reject larger work before allocating
traceback matrices. Set a different `max_cells` only after provisioning memory
and latency accordingly.

## Public objects

- `ScoringParameters`: validated immutable scoring configuration.
- `Alignment`: score, half-open spans, gapped columns, and matched pairs.
- `align`: embedding matrices to exact local alignment.
- `align_scores`: precomputed substitution scores to exact local alignment.
- `rank`: deterministic score-only candidate ranking.
- `similarity_matrix`, `transform_scores`, `normalize_embeddings`.
- `format_alignment`: readable RNA alignment rendering.
- `read_metadata_table`: configurable delimited sequence/structure metadata.

See the
[algorithm and coordinate contract](https://github.com/nicoaira/GINFINITY-SW/blob/main/docs/ALGORITHM.md),
[operations guide](https://github.com/nicoaira/GINFINITY-SW/blob/main/docs/OPERATIONS.md),
and [publishing guide](https://github.com/nicoaira/GINFINITY-SW/blob/main/docs/PUBLISHING.md).

## License

GINFINITY-SW is licensed under the
[PolyForm Noncommercial License 1.0.0](https://github.com/nicoaira/GINFINITY-SW/blob/main/LICENSE).
Commercial use requires a
separate commercial license from the copyright holder.
