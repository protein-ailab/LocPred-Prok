# Dataset Description

## Overview
The dataset used in this study is provided in the file `full_dataset.fasta`. It contains protein sequences annotated with subcellular localization and taxonomic information for prokaryotes (Bacteria and Archaea).

## File Format
The data is stored in standard FASTA format. The header of each entry encodes key metadata separated by vertical bars (`|`).

**Header Structure:**
```text
>Protein_ID|Subcellular_Localization|Taxonomic_Label
```
**Example:**
```text
>A0A0C5CJR8|Extracellular|negative
MSKAKDKAIVSAAQASTAYSQIDSFSHLY...
```
