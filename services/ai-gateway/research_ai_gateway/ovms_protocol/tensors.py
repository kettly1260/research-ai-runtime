"""Canonical <-> KServe v2 tensor conversion.

The gateway's canonical payload is the TFS-era ``instances`` list: one dict of
named, already-padded tensors per batch row.  KServe v2 instead wants a flat
``inputs`` array of named, shaped, typed tensors.  These helpers convert
between the two without hardcoding any model-specific tensor name or shape --
whatever names the caller used are the names that go on the wire.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .base import ProtocolPayloadError


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _shape_of(value: Any) -> List[int]:
    """Return the nested shape of a (possibly nested) tensor literal."""
    shape: List[int] = []
    node = value
    while _is_sequence(node):
        shape.append(len(node))
        if not node:
            break
        node = node[0]
    return shape


def _check_rectangular(value: Any, shape: Sequence[int], index: int, name: str, depth: int = 0) -> None:
    if depth >= len(shape):
        if _is_sequence(value):
            raise ProtocolPayloadError(
                f"instances[{index}].{name} is ragged: nesting is deeper than the first row"
            )
        return
    if not _is_sequence(value) or len(value) != shape[depth]:
        raise ProtocolPayloadError(
            f"instances[{index}].{name} is ragged: expected length {shape[depth]} at depth {depth}"
        )
    for item in value:
        _check_rectangular(item, shape, index, name, depth + 1)


def _flatten(value: Any, out: List[Any]) -> None:
    if _is_sequence(value):
        for item in value:
            _flatten(item, out)
        return
    out.append(value)


def _scalar_datatype(value: Any) -> str:
    # bool is a subclass of int, so it must be checked first.
    if isinstance(value, bool):
        return "BOOL"
    if isinstance(value, int):
        return "INT64"
    if isinstance(value, float):
        return "FP32"
    if isinstance(value, (str, bytes)):
        return "BYTES"
    raise ProtocolPayloadError(f"unsupported tensor element type: {type(value).__name__}")


def tensor_datatype(flat_values: Sequence[Any]) -> str:
    """Infer the KServe ``datatype`` string for a flat tensor payload."""
    if not flat_values:
        raise ProtocolPayloadError("cannot infer a datatype from an empty tensor")
    kinds = set()
    for value in flat_values:
        kinds.add(_scalar_datatype(value))
    if kinds == {"BOOL"}:
        return "BOOL"
    if kinds == {"BYTES"}:
        return "BYTES"
    if kinds <= {"INT64", "FP32"}:
        # Mixed int/float tensors are widened to FP32, matching numpy semantics.
        return "FP32" if "FP32" in kinds else "INT64"
    raise ProtocolPayloadError(f"mixed tensor element types are not supported: {sorted(kinds)}")


def instances_to_inputs(instances: Any) -> List[Dict[str, Any]]:
    """Convert a canonical ``instances`` payload into KServe v2 ``inputs``.

    Tensor names are taken verbatim from the caller.  Ragged rows are rejected
    instead of being silently zero-padded: the gateway pads every request before
    it reaches this layer, so a ragged row means a genuine upstream bug and
    guessing padding here would hide it.
    """
    if not _is_sequence(instances) or len(instances) == 0:
        raise ProtocolPayloadError("payload must contain a non-empty 'instances' list")

    first = instances[0]
    if not isinstance(first, dict):
        raise ProtocolPayloadError("'instances' entries must be objects of named tensors")
    names = list(first.keys())
    if not names:
        raise ProtocolPayloadError("'instances' entries must declare at least one tensor")

    for index, item in enumerate(instances):
        if not isinstance(item, dict) or list(item.keys()) != names:
            raise ProtocolPayloadError(
                f"instances[{index}] declares different tensor names than instances[0]"
            )

    batch = len(instances)
    inputs: List[Dict[str, Any]] = []
    for name in names:
        rows = [item[name] for item in instances]
        row_shape = _shape_of(rows[0])
        for index, row in enumerate(rows):
            _check_rectangular(row, row_shape, index, name)

        flat: List[Any] = []
        _flatten(rows, flat)
        inputs.append(
            {
                "name": name,
                "shape": [batch, *row_shape],
                "datatype": tensor_datatype(flat),
                "data": flat,
            }
        )
    return inputs


def _nest(flat: Sequence[Any], dims: Sequence[int]) -> List[Any]:
    if len(dims) == 1:
        return list(flat)
    step = 1
    for dim in dims[1:]:
        step *= dim
    return [_nest(flat[index * step : (index + 1) * step], dims[1:]) for index in range(dims[0])]


def _materialise(data: Any, shape: Any, batch: int, name: str) -> List[Any]:
    """Turn one KServe output tensor into a list of ``batch`` rows."""
    if _is_sequence(data) and len(data) > 0 and _is_sequence(data[0]):
        # Non-standard but tolerated: the backend already nested the tensor.
        rows = list(data)
        if len(rows) != batch:
            raise ProtocolPayloadError(
                f"output '{name}' returned {len(rows)} rows but the request batch is {batch}"
            )
        return rows

    flat = list(data) if _is_sequence(data) else [data]

    if shape is None:
        if batch <= 0 or len(flat) % batch != 0:
            raise ProtocolPayloadError(
                f"output '{name}' has no usable shape and {len(flat)} values cannot be split into {batch} rows"
            )
        step = len(flat) // batch
        return [flat[index * step : (index + 1) * step] for index in range(batch)]

    try:
        dims = [int(dim) for dim in shape]
    except (TypeError, ValueError) as exc:
        raise ProtocolPayloadError(f"output '{name}' has a non-numeric shape {shape!r}") from exc

    unknown = [index for index, dim in enumerate(dims) if dim < 0]
    if len(unknown) > 1:
        raise ProtocolPayloadError(f"output '{name}' has an ambiguous shape {dims}")
    if unknown:
        known = 1
        for dim in dims:
            if dim >= 0:
                known *= dim
        if known <= 0 or len(flat) % known != 0:
            raise ProtocolPayloadError(
                f"output '{name}' shape {dims} does not match {len(flat)} returned values"
            )
        dims[unknown[0]] = len(flat) // known

    declared = 1
    for dim in dims:
        declared *= dim
    if declared != len(flat):
        raise ProtocolPayloadError(
            f"output '{name}' shape {dims} declares {declared} values but {len(flat)} were returned"
        )

    if not dims:
        if batch != 1:
            raise ProtocolPayloadError(f"output '{name}' is scalar but the request batch is {batch}")
        return [flat[0]]
    if len(dims) == 1:
        # A single flat vector for a single-row request is a legitimate shape.
        if batch == 1:
            return [flat]
        raise ProtocolPayloadError(
            f"output '{name}' has shape {dims} but the request batch is {batch}"
        )
    if dims[0] != batch:
        raise ProtocolPayloadError(
            f"output '{name}' leading dimension {dims[0]} does not match the request batch {batch}"
        )
    return _nest(flat, dims)


#: Output datatypes the gateway knows how to turn into float vectors.  A
#: backend that declares anything else is reporting a contract violation, not a
#: vector, so it fails closed instead of being coerced downstream.
_NUMERIC_DATATYPES = {
    "BOOL",
    "INT8",
    "UINT8",
    "INT16",
    "UINT16",
    "INT32",
    "UINT32",
    "INT64",
    "UINT64",
    "FP16",
    "BF16",
    "FP32",
    "FP64",
}


def _check_output_datatype(name: str, datatype: Any) -> None:
    if datatype is None:
        return
    if not isinstance(datatype, str) or datatype.upper() not in _NUMERIC_DATATYPES:
        raise ProtocolPayloadError(
            f"output '{name}' declares unsupported datatype {datatype!r}"
        )


def _check_output_sample(name: str, rows: Sequence[Any]) -> None:
    """Cheap numeric sanity check on the first returned scalar."""
    if not rows:
        return
    probe: Any = rows[0]
    while _is_sequence(probe):
        if len(probe) == 0:
            return
        probe = probe[0]
    try:
        _scalar_datatype(probe)
    except ProtocolPayloadError as exc:
        raise ProtocolPayloadError(f"output '{name}' returned non-numeric data") from exc


def outputs_to_predictions(outputs: Any, batch: int) -> List[Any]:
    """Normalise KServe v2 ``outputs`` into the canonical ``predictions`` list.

    * single output  -> one row per batch entry (identical to TFS ``predictions``)
    * several outputs -> one dict per batch entry, keyed by output name
    """
    if not _is_sequence(outputs) or len(outputs) == 0:
        raise ProtocolPayloadError("kserve response is missing 'outputs'")

    tensors: List[Tuple[str, List[Any]]] = []
    for index, output in enumerate(outputs):
        if not isinstance(output, dict):
            raise ProtocolPayloadError(f"outputs[{index}] is not an object")
        name = output.get("name") or f"output_{index}"
        if "data" not in output or output.get("data") is None:
            raise ProtocolPayloadError(f"outputs[{index}] ('{name}') is missing 'data'")
        _check_output_datatype(name, output.get("datatype"))
        rows = _materialise(output.get("data"), output.get("shape"), batch, name)
        _check_output_sample(name, rows)
        tensors.append((name, rows))

    if len(tensors) == 1:
        return list(tensors[0][1])

    return [
        {name: rows[batch_index] for name, rows in tensors}
        for batch_index in range(batch)
    ]
