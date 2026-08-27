from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ExampleDataModel(BaseModel):
    patient_id: Annotated[
        str,
        Field(
            pattern=r"^PT-\d{6}$",
            description="Patient identifier, fixed format PT-000001.",
        ),
    ]
    sex: Annotated[
        Literal["F", "M"],
        Field(description="Administrative sex."),
    ]
    age_at_diagnosis: Annotated[
        int,
        Field(ge=0, le=110, description="Age in completed years at diagnosis."),
    ]
    bmi: Annotated[
        float,
        Field(ge=10.0, le=70.0, description="Body mass index (kg/m^2)."),
    ]
    charlson_ci: Annotated[
        int,
        Field(ge=0, le=20, description="Charlson Comorbidity Index score."),
    ]
    disease_stage: Annotated[
        Literal["I", "II", "III", "IV"],
        Field(description="Disease stage at diagnosis."),
    ]
