

if __name__ == "__main__":
    from arguments import CreateModelArgs
    import conv_autoencoder as ae

    name_model = "autoencoder_emocional_dinamico"

    args = CreateModelArgs().parse()

    ae.checkModel(args.dataroot, name_model=name_model, height=args.height, width=args.width)