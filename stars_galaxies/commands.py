def train() -> None:
    """Run model training."""
    pass


def infer() -> None:
    """Run inference on new data."""
    pass


def export_onnx() -> None:
    """Export trained model to ONNX."""
    pass


def export_trt() -> None:
    """Export ONNX model to TensorRT."""
    pass


def main() -> None:
    import fire

    fire.Fire(
        {
            "train": train,
            "infer": infer,
            "export_onnx": export_onnx,
            "export_trt": export_trt,
        }
    )


if __name__ == "__main__":
    main()