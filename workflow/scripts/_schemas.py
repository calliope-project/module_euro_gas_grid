"""Schemas for key files."""

from pandera import pandas as pa
from pandera.typing import Series
from pandera.typing.geopandas import GeoSeries

ISO3_RE = r"^[A-Z]{3}$"

class LandSchema(pa.DataFrameModel):
    class Config:
        coerce = True
        strict = "filter"

    feature_class: Series[str] = pa.Field(eq="Land")
    "Must be 'Land'."
    geometry: GeoSeries
    "Land polygons."

class PipelineSchema(pa.DataFrameModel):
    class Config:
        coerce = True
        strict = "filter"
    name: Series[str]
    "Pipeline name."
    start_point: GeoSeries
    "Pipeline start point."
    start_country_id: Series[str] = pa.Field(str_length=3)
    "ISO 3 code of the country in the start point."
    end_point: GeoSeries
    "Pipeline end point."
    end_country_id: Series[str] = pa.Field(str_length=3)
    "ISO 3 code of the country in the start point."
    diameter_mm: Series[float]
    "Pipeline diameter."
    diameter_method: Series[str]
    "Diameter estimation metadata."
    max_cap_M_m3_per_d: Series[float]
    "Max capacity estimate."
    max_cap_method: Series[str]
    "Max capacity metadata (used to select formulae)."
    max_pressure_bar: Series[float]
    "Max pressure (used for sectioning)."
    is_bothDirection: Series[bool]
    "Pipeline direction."
    is_offshore: Series[bool]
    "Flag offshore pipelines (outside of country landmass)."
    geometry: GeoSeries
    "Lines."
