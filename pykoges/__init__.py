from pykoges import __codingbook, __koges, __learn, stats, utils

codingbook = __codingbook.codingbook

koges = __koges.kogesclass
koges.Variables = __koges.Variables

model = __learn.modelclass


__all__ = [
    "codingbook",
    "koges",
    "stats",
    "utils",
    "model",
]

del __codingbook, __koges, __learn
