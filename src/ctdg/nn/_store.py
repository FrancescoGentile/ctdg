# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from typing import Generic, Literal, TypeAlias, TypeVar

import torch
from torch import Tensor

from ctdg.data import Stream

_T = TypeVar("_T", bound=Stream)


class StreamStore(Generic[_T]):
    """A class to store streams for each node."""

    ConflictMode: TypeAlias = Literal["replace", "append", "error"]
    """Specifies the conflict resolution mode for storing events.

    - `'replace'`: Replace the stored events with the new ones.
    - `'append'`: Append the new events to the stored ones.
    - `'error'`: Raise an error if events are assigned to a node that already has
      stored events.
    """

    def __init__(self, num_nodes: int, conflict_mode: ConflictMode = "append") -> None:
        """Initializes the event store.

        Args:
            num_nodes: The number of nodes for which to store stream.
            conflict_mode: What to do if stream are assigned to a node that already has
                stored stream.
        """
        self._num_nodes = num_nodes
        self._conflict_mode: StreamStore.ConflictMode = conflict_mode
        self._store: list[_T | None] = [None] * num_nodes

    def clear(self, idx: Tensor | None = None) -> None:
        """Removes the stream stored for the given nodes.

        Args:
            idx: The indices of the nodes for which to remove the stored stream.
                If `None`, all stored stream are removed.
        """
        if idx is None:
            self._store = [None] * self._num_nodes
        else:
            for i in idx.tolist():
                self._store[i] = None

    def store(self, idx: Tensor, streams: _T) -> None:
        """Stores the given stream.

        Args:
            idx: The indices of the nodes for which to store the streams. This must be
                a 1D tensor with the same length as `streams`.
            streams: The streams to store.
        """
        if len(idx) != len(streams):
            msg = "The number of indices must match the number of stream."
            raise ValueError(msg)

        n_id, perm = torch.sort(idx)
        n_id, count = torch.unique_consecutive(n_id, return_counts=True)
        for i, c in zip(n_id.tolist(), perm.split(count.tolist()), strict=True):
            if self._store[i] is not None:
                match self._conflict_mode:
                    case "replace":
                        self._store[i] = streams[c]
                    case "append":
                        cls = self._store[i].__class__
                        self._store[i] = cls.cat([self._store[i], streams[c]])
                    case "error":
                        msg = f"Streams already stored for node {i}."
                        raise RuntimeError(msg)
            else:
                self._store[i] = streams[c]

    def retrieve(self, idx: Tensor, *, clear: bool = False) -> tuple[Tensor, _T] | None:
        """Retrieves the stored stream for the given nodes.

        !!! note

            The returned stream that refer to the same node are contiguous and sorted
            in the same order as they were stored. This means that globally the stream
            may not be sorted by timestamp.

        Args:
            idx: The indices of the nodes for which to retrieve the stored stream.
            clear: Whether to remove the stored stream after retrieving them.

        Returns:
            A tuple containing the indices of the nodes for which stream were retrieved
            and the retrieved stream. If no stream were retrieved, `None` is returned.
        """
        idx_list: list[int] = idx.tolist()
        stream, indices = [], []
        for i in idx_list:
            data = self._store[i]
            if data is not None:
                stream.append(data)
                idx = torch.full((len(data),), i, device=idx.device, dtype=idx.dtype)
                indices.append(idx)

                if clear:
                    self._store[i] = None

        if len(stream) == 0:
            return None

        cls = stream[0].__class__
        return torch.cat(indices), cls.cat(stream)
