"""Exact token semantics are checked before any canonical numeric pin is issued."""
from decimal import Decimal, localcontext
import json

import pytest

from hermes_lcm.literal_record import validate_row
from .test_literal_record import engine, literal, row


@pytest.mark.parametrize("field", ["value", "condition"])
@pytest.mark.parametrize("token", ["1e-400", "-1e-400", "0.100000000000000000001",
    "9007199254740991.1", "-9007199254740991.1", "9007199254740991.0000000000000000000001",
    "9007199254740992", "1e400", "1e99999999999999999999999999"])
def test_original_unsupported_number_abstains_even_under_low_decimal_precision(engine, field, token):
    data = literal()
    if field == "condition":
        data["scope"]["conditions"] = [{"namespace": "env", "id": "limit", "value": "NUMBER_TOKEN"}]
    else:
        data["value"] = "NUMBER_TOKEN"
    text = json.dumps(data).replace('"NUMBER_TOKEN"', token)
    source = row(engine, text)
    with localcontext() as context:
        context.prec = 3
        assert validate_row(source, 0, len(text)) is None
    assert source["content"] == text


@pytest.mark.parametrize("field", ["value", "condition"])
@pytest.mark.parametrize("token", ["0.1", "1.25", "-0.0", "1e-200", "9007199254740991",
                                   "9007199254740991.0", "-9007199254740991.0"])
def test_representable_number_uses_existing_canonical_json_without_changing_bytes(engine, field, token):
    data = literal()
    if field == "condition":
        data["scope"]["conditions"] = [{"namespace": "env", "id": "limit", "value": "NUMBER_TOKEN"}]
    else:
        data["value"] = "NUMBER_TOKEN"
    text = json.dumps(data).replace('"NUMBER_TOKEN"', token)
    source = row(engine, text)
    with localcontext() as context:
        context.prec = 3
        value = validate_row(source, 0, len(text))
    assert value is not None
    pins = dict(value.coordinate_pins)
    number = json.loads(pins["value" if field == "value" else "scope"], parse_float=Decimal, parse_int=Decimal)
    if field == "condition":
        number = number["conditions"][0]["value"]
    assert number == Decimal(token)
    assert value.source_bytes == text.encode()
    assert value.record_span == (0, len(text))
