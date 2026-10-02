from src.train_zoo import FAULTS, model_spec


def test_model_zoo_spec_is_balanced_by_design():
    specs = [model_spec(i) for i in range(1, 101)]
    assert len(specs) == 100
    assert {spec.fault for spec in specs} == set(FAULTS)
    assert all(0 <= spec.severity <= 1 for spec in specs)


def test_model_zoo_specs_are_reproducible():
    assert model_spec(17) == model_spec(17)
    assert model_spec(17).seed == 59
