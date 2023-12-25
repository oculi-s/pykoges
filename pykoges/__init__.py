from .datatype import Answer, Patient, Patients, Question, Questions
from .codingbook import read
from .koges import *
from .stats import *

__all__ = [
    "Answer",
    "Patient",
    "Patients",
    "Question",
    "Questions",
    #
    "read",
    "summary",
    "koges",
    "stats",
]
codingbook.__all__ = ["read"]
koges.__all__ = [
    "read",
    "convert",
    "drop",
    "split_data",
]
stats.__all__ = [
    "normality",
    "homogenity",
    "split",
    "summary",
    "anova",
    "t_test",
    "boxplot",
]
