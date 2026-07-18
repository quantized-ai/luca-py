"""Media source tagged union — used by ImageBlock / AudioBlock / FileBlock."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class MediaURL(BaseModel):
    kind: Literal["url"] = "url"
    url: str
    media_type: str | None = None

    model_config = ConfigDict(extra="forbid")


class MediaBase64(BaseModel):
    kind: Literal["base64"] = "base64"
    data: str
    media_type: str

    model_config = ConfigDict(extra="forbid")


class MediaFileId(BaseModel):
    kind: Literal["file"] = "file"
    file_id: str
    media_type: str | None = None

    model_config = ConfigDict(extra="forbid")


MediaSource = Annotated[
    Union[MediaURL, MediaBase64, MediaFileId],
    Field(discriminator="kind"),
]
