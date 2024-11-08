from enum import Enum


class SKIMode(Enum):
    Q = "q"  # RAG (retrieval)

    QC = "qc"  # RAG (retrieval), SFT
    QCASM = "qc-asm"  # RAG (retrieval), SFT

    QA = "qa"  # SFT, CPT
    QAASM = "qa-asm"  # CPT

    QCA = "qca"  # SFT, CPT
    QCAASM = "qca-asm"  # CPT
