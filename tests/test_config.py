import pytest
import numpy as np
from app import predict, NotANumber

input_data = {
    "incorrect_values": {"Temperature": 30, "Oxygen": 15, "Humidity": 'as'},
    "correct_values": {"Temperature": 32, "Oxygen": 15, "Humidity": 15}
}

def test_predict_correct_values():
    data = np.array(list(input_data["correct_values"].values()), dtype=float).reshape(1, -1)
    result = predict(data)
    assert isinstance(result, str)  # adjust the assertion based on your function's expected behavior

def test_predict_incorrect_values():
    data = np.array(list(input_data["incorrect_values"].values()), dtype=object).reshape(1, -1)
    with pytest.raises(NotANumber):
        predict(data)
