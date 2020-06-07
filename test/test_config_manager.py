from modules.config.config_manager import ConfigurationManager

def test_model_name():
    json={"model_name":"vgg"}
    config=ConfigurationManager(json)
    assert config.model_name=="vgg"

def test_num_classes():
    json={"num_classes":"10"}
    config=ConfigurationManager(json)
    assert config.num_classes == 10

