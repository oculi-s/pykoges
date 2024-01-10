from pykoges import __codingbook, __koges, __learn, stats, utils, __dicom

codingbook = __codingbook.codingbook

koges = __koges.kogesclass
koges.Variables = __koges.Variables
dicom = __dicom.Dicom

model = __learn.modelclass


__all__ = ["codingbook", "koges", "stats", "utils", "model", "dicom"]

del __codingbook, __koges, __learn, __dicom
