from datetime import date
from typing import Optional, Any, Dict, AnyStr, List

import json
import nest_asyncio
import pandas as pd

from mosqlient import types
from mosqlient.client import Mosqlient, Client
from mosqlient.registry import schema

nest_asyncio.apply()


class Model(types.Model):
    client: Optional[Client] = None
    _schema: schema.Model

    def __init__(
        self,
        id: int,
        repository: str,
        disease: str,
        category: str,
        adm_level: int,
        time_resolution: str,
        predictions_count: int,
        active: bool,
        created_at: date,
        last_update: date,
        description: Optional[str] = "",
        sprint: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._schema = schema.Model(
            id=id,
            repository=repository,
            description=description,
            disease=disease,
            category=category,
            adm_level=adm_level,
            time_resolution=time_resolution,
            sprint=sprint,
            predictions_count=predictions_count,
            active=active,
            created_at=created_at,
            last_update=last_update,
        )

    def __repr__(self) -> str:
        return self.repository

    @classmethod
    def get(cls, api_key: str, **kwargs):
        client = Mosqlient(x_uid_key=api_key)
        params = schema.ModelGETParams(**kwargs)
        return list(cls(**item) for item in client.get(params))

    def predictions(self, api_key: str, **kwargs):
        return Prediction.get(api_key=api_key, model_id=self.id, **kwargs)

    @property
    def id(self) -> int:
        return self._schema.id

    @property
    def repository(self) -> str:
        return self._schema.repository

    @property
    def description(self) -> Optional[str]:
        return self._schema.description

    @property
    def disease(self) -> str:
        return self._schema.disease

    @property
    def category(self) -> str:
        return self._schema.category

    @property
    def adm_level(self) -> int:
        return self._schema.adm_level

    @property
    def time_resolution(self) -> str:
        return self._schema.time_resolution

    @property
    def sprint(self) -> int | None:
        return self._schema.sprint

    @property
    def predictions_count(self) -> int:
        return self._schema.predictions_count

    @property
    def active(self) -> bool:
        return self._schema.active

    @property
    def created_at(self) -> date:
        return self._schema.created_at

    @property
    def last_update(self) -> date:
        return self._schema.last_update


class Prediction(types.Model):
    client: Optional[Client] = None
    model: Model
    _schema: schema.Prediction

    def __init__(
        self,
        id: int,
        model: Model | dict,
        commit: types.Commit,
        case_definition: str,
        published: bool,
        created_at: date,
        description: types.Description = "",
        start: Optional[date] = None,
        end: Optional[date] = None,
        scores: Optional[Dict[str, float]] = None,
        adm_0: Optional[str] = None,
        adm_1: Optional[int] = None,
        adm_2: Optional[int] = None,
        adm_3: Optional[int] = None,
        data: Optional[types.PredictionData] = None,
        client: Optional[Client] = None,
        **kwargs,
    ):
        if isinstance(model, dict):
            model = Model(**model)

        kwargs["model"] = model
        kwargs["client"] = client
        super().__init__(**kwargs)

        self.client = client

        _data = []
        if data:
            if isinstance(data, str):
                try:
                    loaded = json.loads(data)
                    _data = [schema.PredictionDataRow(**d) for d in loaded]
                except json.decoder.JSONDecodeError:
                    raise ValueError("str `data` must be JSON serializable")
            elif isinstance(data, pd.DataFrame):
                _data = [
                    schema.PredictionDataRow(**d)
                    for d in data.to_dict(orient="records")
                ]
            elif isinstance(data, list):
                _data = [schema.PredictionDataRow(**d) for d in data]

        self.model = model
        self._schema = schema.Prediction(
            id=id,
            model=model._schema,
            commit=commit,
            description=description,
            case_definition=case_definition,
            published=published,
            created_at=created_at,  # type: ignore
            start=start,  # type: ignore
            end=end,  # type: ignore
            scores=scores or {},
            adm_0=adm_0,
            adm_1=adm_1,
            adm_2=adm_2,
            adm_3=adm_3,
            data=_data,
        )

    def __repr__(self) -> str:
        return f"Prediction <{self.id}>"

    @classmethod
    def get(cls, api_key: str, **kwargs):
        client = Mosqlient(x_uid_key=api_key)
        params = schema.PredictionGETParams(**kwargs)
        return list(cls(**item, client=client) for item in client.get(params))

    @classmethod
    def post(cls, api_key: str, **kwargs):
        client = Mosqlient(x_uid_key=api_key)
        params = schema.PredictionPOSTParams(**kwargs)
        res = client.post(params)
        data = json.loads(res.text)

        if "id" in data:
            predictions = cls.get(api_key=api_key, id=data["id"])
            if predictions:
                return predictions[0]

        return cls(**data, client=client)

    def delete(self, api_key: str):
        if not self.id:
            raise ValueError("Cannot delete a prediction that has no ID.")
        return self.delete_by_id(api_key=api_key, id=self.id)

    @classmethod
    def delete_by_id(cls, api_key: str, id: int):
        client = Mosqlient(x_uid_key=api_key)
        params = schema.PredictionDELETEParams(id=id)
        return client.delete(params)

    @property
    def id(self) -> types.ID | None:
        return self._schema.id

    @property
    def description(self) -> types.Description:
        return self._schema.description

    @property
    def commit(self) -> types.Commit:
        return self._schema.commit

    @property
    def data(self) -> List[Dict[AnyStr, Any]]:
        if not self._schema.data and self.client and self.id:
            params = schema.PredictionDataGETParams(id=self.id)
            raw_data = self.client.get(params)
            self._schema.data = [
                schema.PredictionDataRow(**d) for d in raw_data
            ]

        return [row.dict() for row in (self._schema.data or [])]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)

    @property
    def case_definition(self) -> str | None:
        return self._schema.case_definition

    @property
    def published(self) -> bool:
        return self._schema.published

    @property
    def start(self) -> date | None:
        return self._schema.start  # type: ignore

    @property
    def end(self) -> date | None:
        return self._schema.end  # type: ignore

    @property
    def scores(self) -> Dict[str, float]:
        return self._schema.scores or {}

    @property
    def created_at(self) -> date:
        return self._schema.created_at  # type: ignore

    @property
    def adm_0(self) -> str | None:
        return self._schema.adm_0

    @property
    def adm_1(self) -> int | None:
        return self._schema.adm_1

    @property
    def adm_2(self) -> int | None:
        return self._schema.adm_2
