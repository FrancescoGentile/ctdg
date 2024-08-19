# Copyright 2024 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, TypeAlias

import torch
from torch import Tensor

from ctdg.structures import Events


class EventStore:
    """A class to store events for each node.

    This class can be used to store for each node the events it has been involved in
    and retrieve them later.
    """

    ConflictMode: TypeAlias = Literal["replace", "append", "error"]
    """Specifies the conflict resolution mode for storing events.

    - `'replace'`: Replace the stored events with the new ones.
    - `'append'`: Append the new events to the stored ones.
    - `'error'`: Raise an error if events are assigned to a node that already has
      stored events.
    """

    def __init__(
        self,
        num_nodes: int,
        conflict_mode: ConflictMode = "append",
    ) -> None:
        """Initializes the event store.

        Args:
            num_nodes: The number of nodes for which to store events.
            conflict_mode: What to do if events are assigned to a node that already has
                stored events.
        """
        self._num_nodes = num_nodes
        self._conflict_mode: EventStore.ConflictMode = conflict_mode
        self._store: list[Events | None] = [None] * num_nodes

    def clear(self, idx: Tensor | None = None) -> None:
        """Removes the events stored for the given nodes.

        Args:
            idx: The indices of the nodes for which to remove the stored events.
                If `None`, all stored events are removed.
        """
        if idx is None:
            self._store = [None] * self._num_nodes
        else:
            for i in idx.tolist():
                self._store[i] = None

    def store(self, idx: Tensor, events: Events) -> None:
        """Stores the given events.

        Args:
            idx: The indices of the nodes for which to store the events. This must be
                a 1D tensor with the same length as `events`.
            events: The events to store.
        """
        if len(idx) != len(events):
            msg = "The number of indices must match the number of events."
            raise ValueError(msg)

        n_id, perm = torch.sort(idx)
        n_id, count = torch.unique_consecutive(n_id, return_counts=True)
        for i, c in zip(n_id.tolist(), perm.split(count.tolist()), strict=True):
            if self._store[i] is not None:
                match self._conflict_mode:
                    case "replace":
                        self._store[i] = events[c]
                    case "append":
                        self._store[i] = Events.cat([self._store[i], events[c]])
                    case "error":
                        msg = f"Events already stored for node {i}."
                        raise RuntimeError(msg)
            else:
                self._store[i] = events[c]

    def retrieve(self, idx: Tensor, *, clear: bool = False) -> tuple[Tensor, Events]:
        """Retrieves the stored events for the given nodes.

        !!! note

            The returned events that refer to the same node are contiguous and sorted
            in the same order as they were stored. This means that globally the events
            may not be sorted by timestamp.

        Args:
            idx: The indices of the nodes for which to retrieve the stored events.
            clear: Whether to remove the stored events after retrieving them.

        Returns:
            A tuple containing the indices of the nodes for which events were retrieved
            and the retrieved events.
        """
        idx_list: list[int] = idx.tolist()
        events, indices = [], []
        for i in idx_list:
            data = self._store[i]
            if data is not None:
                events.append(data)
                idx = torch.full((len(data),), i, device=idx.device, dtype=idx.dtype)
                indices.append(idx)

                if clear:
                    self._store[i] = None

        if len(events) == 0:
            indices = torch.tensor([], device=idx.device, dtype=idx.dtype)
            events = Events.empty(device=idx.device)
            return indices, events

        return torch.cat(indices), Events.cat(events)
