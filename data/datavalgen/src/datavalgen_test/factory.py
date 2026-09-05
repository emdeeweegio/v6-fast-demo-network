import itertools
from typing import Literal

from datavalgen.factory import BaseDataModelFactory
from polyfactory.decorators import post_generated

from .model import ExampleDataModel


class ExampleDataFactory(BaseDataModelFactory):
    """Factory for generating fake patient records."""

    __model__ = ExampleDataModel

    _id_counter = itertools.count(1)

    @classmethod
    def patient_id(cls) -> str:
        # Sequential, fixed-format IDs so every generated patient follows the
        # same PT-###### shape the model validates against.
        return f"PT-{next(cls._id_counter):06d}"

    @classmethod
    def sex(cls) -> Literal["female", "male"]:
        return "female" if cls.__random__.random() < 0.5 else "male"

    @classmethod
    def age_at_diagnosis(cls) -> int:
        # Diagnosis skews older; clamp to plausible bounds.
        value = cls.__random__.gauss(65, 12)
        return int(min(max(round(value), 18), 100))

    @post_generated
    @classmethod
    def bmi(cls, sex: Literal["female", "male"]) -> float:
        mean_bmi = {"female": 26.5, "male": 27.5}[sex]
        value = cls.__random__.gauss(mean_bmi, 4.5)
        return round(min(max(value, 15.0), 55.0), 1)

    @post_generated
    @classmethod
    def charlson_ci(cls, age_at_diagnosis: int) -> int:
        # Comorbidity burden trends upward with age.
        mean_ci = max(0.0, (age_at_diagnosis - 40) / 15)
        value = cls.__random__.gauss(mean_ci, 1.5)
        return int(min(max(round(value), 0), 20))

    @classmethod
    def disease_stage(cls) -> Literal["I", "II", "III", "IV"]:
        roll = cls.__random__.random()
        if roll < 0.30:
            return "I"
        if roll < 0.60:
            return "II"
        if roll < 0.85:
            return "III"
        return "IV"
