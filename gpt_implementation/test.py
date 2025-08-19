from model import *

if __name__ == "__main__":
    model_config = DecoderConfig(
        n_layers = 6,
        d_model = 8,
        n_heads = 4,
        d_head = 4,
        alphabet_size = 32
    )
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print(device)
    print(t.cuda.get_device_name())

    model = model_config.get_model()

    model.to(device)

    for name, param in model.named_parameters():
        print(name, param.get_device(), param.requires_grad)

    test_arr = t.randint(0, 31, (64, 256)).to(device)
    """
    test_arr = t.randn((64, 256, 16)).to(device)

    tmp_attention_block = AttentionBlock(
        8, 8, 16, 8 
    ).to(device)

    print(test_arr.get_device())
    """

    print(model(test_arr)[0])